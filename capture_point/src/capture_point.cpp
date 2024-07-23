#include <memory>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cmath>
#include <list>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "biped_bringup/msg/stamped_bool.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "rosgraph_msgs/msg/clock.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "eigen3/Eigen/Dense"

#include "trajectory_optimization.hpp"

using namespace std::placeholders;
using namespace std::chrono_literals;

class CapturePoint : public rclcpp::Node
{

public:
    CapturePoint() : Node("cp_node")
    {
        robot_params_.robot_height = this->declare_parameter<double>("robot_height",  0.54);
        robot_params_.t_step = this->declare_parameter<double>("t_step", 0.25);
        robot_params_.dt_ctrl = this->declare_parameter<double>("ctrl_time_sec", 0.01);
        robot_params_.duration_init_traj = this->declare_parameter<double>("duration_init_traj", 3.0);
        robot_params_.safety_radius_CP = this->declare_parameter<double>("safety_radius_CP", 1.0);
        robot_params_.T_contact_ignore = this->declare_parameter<double>("T_contact_ignore", 0.1);
        robot_params_.omega = sqrt(9.81 / robot_params_.robot_height);
        robot_params_.offset_baselink_cog_x = this->declare_parameter<double>("offset_baselink_cog_x", 0.0);
        robot_params_.offset_baselink_cog_y = this->declare_parameter<double>("offset_baselink_cog_y", 0.0);
        robot_params_.offset_baselink_cog_z = this->declare_parameter<double>("offset_baselink_cog_z", 0.0);
        robot_params_.time_no_feet_in_contact = this->declare_parameter<double>("time_no_feet_in_contact", 0.2);
        robot_params_.foot_separation = this->declare_parameter<double>("foot_separation", 0.1);
        robot_params_.swing_x_safe_box_min = this->declare_parameter<double>("swing_x_safe_box_min", -0.2);
        robot_params_.swing_x_safe_box_max = this->declare_parameter<double>("swing_x_safe_box_max", 0.2);
        robot_params_.swing_y_safe_box_min = this->declare_parameter<double>("swing_y_safe_box_min", -0.2);
        robot_params_.swing_y_safe_box_max = this->declare_parameter<double>("swing_y_safe_box_max", 0.2);
        robot_params_.swing_z_safe_box_min = this->declare_parameter<double>("swing_z_safe_box_min", 0.0);
        robot_params_.swing_z_safe_box_max = this->declare_parameter<double>("swing_z_safe_box_max", 0.2);
        robot_params_.walk_slow = this->declare_parameter<bool>("walk_slow", true);

        state_ = "INIT";
        initialization_done_ = false;
        t_init_traj_ = 0.0;
        start_cmd_line_ = false;

        walk_slow_ = robot_params_.walk_slow;

        if (walk_slow_ == true)
        {
            swing_foot_traj_ = OptimizerTrajectory(robot_params_.dt_ctrl, robot_params_.t_step);
        } else {
            swing_foot_traj_ = OptimizerTrajectory(robot_params_.dt_ctrl, robot_params_.t_step);
        }

        r_foot_frame_id_ = this->declare_parameter<std::string>("r_foot_frame_id", "R_FOOT");
        l_foot_frame_id_ = this->declare_parameter<std::string>("l_foot_frame_id", "L_FOOT");
        r_foot_urdf_frame_id_ = this->declare_parameter<std::string>("r_foot_urdf_frame_id", "R_FOOT");
        l_foot_urdf_frame_id_ = this->declare_parameter<std::string>("l_foot_urdf_frame_id", "L_FOOT");
        base_link_frame_id_ = this->declare_parameter<std::string>("base_link_frame_id", "base_link");
        mode_ = this->declare_parameter<std::string>("mode", "ONE_FOOT_SWING");
        if (mode_ != "WALK" && mode_ != "ONE_FOOT_SWING") {
            RCLCPP_ERROR(this->get_logger(), "Mode not supported");
            throw std::runtime_error("Mode not supported");
        }

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        pub_body_trajectory_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/body_trajectories", 10);

        pub_markers_foot_traj_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_traj_feet", 10);
        pub_markers_foot_traj_actual_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_actual_traj_feet", 10);

        pub_markers_safety_circle_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_safety_circle", 10);
        pub_marker_next_footstep_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_next_footstep", 10);
        pub_marker_dcm_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_dcm", 10);
        pub_marker_desired_dcm_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_desired_dcm", 10);
        pub_marker_stance_foot_BF_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_stance_foot_BF", 10);
        pub_marker_swing_foot_BF_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_swing_foot_BF", 10);

        pub_desired_dcm_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("~/desired_dcm", 10);
        pub_predicted_dcm_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("~/predicted_dcm", 10);

        pub_desired_left_contact_ = this->create_publisher<biped_bringup::msg::StampedBool>("~/desired_left_contact", 10);
        pub_desired_right_contact_ = this->create_publisher<biped_bringup::msg::StampedBool>("~/desired_right_contact", 10);

        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", 10, std::bind(&CapturePoint::odometry_callback, this, _1));

        contact_right_sub_ = this->create_subscription<biped_bringup::msg::StampedBool>(
            "~/contact_foot_right", 10, std::bind(&CapturePoint::contact_right_callback, this, _1));

        contact_left_sub_ = this->create_subscription<biped_bringup::msg::StampedBool>(
            "~/contact_foot_left", 10, std::bind(&CapturePoint::contact_left_callback, this, _1));

        vel_cmd_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            "~/vel_cmd", 10, std::bind(&CapturePoint::vel_cmd_cb, this, _1));

        e_stop_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "~/e_stop", 10, std::bind(&CapturePoint::e_stop_cb, this, _1));
        e_stop_ = false;

        std::chrono::duration<double> period = robot_params_.dt_ctrl * 1s;
        timer_ = rclcpp::create_timer(this, this->get_clock(), period, std::bind(&CapturePoint::timer_callback, this));
    }

private:
    void set_starting_to_walk_params(Eigen::Vector3d swing_foot_STF)
    {
        foot_right_contact_ = false;
        foot_left_contact_ = false;

        time_since_last_step_ = robot_params_.t_step / 2;
        remaining_time_in_step_ = robot_params_.t_step - time_since_last_step_;
        timeout_for_no_feet_in_contact_ = 0;

        swing_foot_position_beginning_of_step_STF_ = swing_foot_STF;
        foot_traj_list_STF_.clear(); // used for markers
        foot_actual_traj_list_STF_.clear(); // used for markers
        start_opt_pos_swing_foot_ = swing_foot_position_beginning_of_step_STF_;
        start_opt_vel_swing_foot_ = Eigen::Vector3d::Zero();
        swing_foot_traj_.set_initial_pos_vel(start_opt_pos_swing_foot_, start_opt_vel_swing_foot_);
        dcm_at_step_STF_ << 0.0, 0.0, 0.0;
    }

    void set_position_limits_for_foot_in_optimization(std::string swing_foot_name)
    {
        Eigen::Vector2d swing_x_safe_box_min_max, swing_y_safe_box_min_max,swing_z_safe_box_min_max;
        swing_x_safe_box_min_max << robot_params_.swing_x_safe_box_min, robot_params_.swing_x_safe_box_max;
        swing_z_safe_box_min_max << robot_params_.swing_z_safe_box_min, robot_params_.swing_z_safe_box_max;

        if (swing_foot_name == r_foot_frame_id_)
        {
            swing_y_safe_box_min_max << robot_params_.swing_y_safe_box_min, -robot_params_.foot_separation * 0.5;
        } else{
            swing_y_safe_box_min_max << robot_params_.foot_separation * 0.5, robot_params_.swing_z_safe_box_max;
        }
        swing_foot_traj_.set_position_limits(swing_x_safe_box_min_max, swing_y_safe_box_min_max, swing_z_safe_box_min_max);
    }

    void e_stop_cb(const std_msgs::msg::Bool::SharedPtr msg)
    {
        e_stop_ = msg->data;
    }

    void odometry_callback(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        base_link_odom_.stamp = rclcpp::Time(msg->header.stamp);
        base_link_odom_.position(0) = msg->pose.pose.position.x;
        base_link_odom_.position(1) = msg->pose.pose.position.y;
        base_link_odom_.position(2) = msg->pose.pose.position.z;
        base_link_odom_.orientation.w() = msg->pose.pose.orientation.w;
        base_link_odom_.orientation.x() = msg->pose.pose.orientation.x;
        base_link_odom_.orientation.y() = msg->pose.pose.orientation.y;
        base_link_odom_.orientation.z() = msg->pose.pose.orientation.z;
        base_link_odom_.linear_velocity(0) = msg->twist.twist.linear.x;
        base_link_odom_.linear_velocity(1) = msg->twist.twist.linear.y;
        base_link_odom_.linear_velocity(2) = msg->twist.twist.linear.z;
        base_link_odom_.angular_velocity(0) = msg->twist.twist.angular.x;
        base_link_odom_.angular_velocity(1) = msg->twist.twist.angular.y;
        base_link_odom_.angular_velocity(2) = msg->twist.twist.angular.z;
    }

    void timer_callback()
    {
        if (base_link_odom_.stamp == rclcpp::Time(0, 0, RCL_ROS_TIME)) {
            RCLCPP_INFO(this->get_logger(), "Waiting for odometry...");
            return;
        }
        if ((this->get_clock()->now() - base_link_odom_.stamp).seconds() > 0.1) {
            RCLCPP_ERROR(this->get_logger(), "Odometry is too old");
            return;
        }

        if (!tf_buffer_->canTransform(base_link_frame_id_, r_foot_frame_id_, tf2::TimePointZero) ||
            !tf_buffer_->canTransform(base_link_frame_id_, l_foot_frame_id_, tf2::TimePointZero)) {
            RCLCPP_INFO(this->get_logger(), "Waiting for TFs for R_FOOT and L_FOOT...");
            return;
        }

        if (e_stop_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "E-STOP is active");
            return;
        }

        if (foot_left_contact_ == false && foot_right_contact_ == false) {
            timeout_for_no_feet_in_contact_ -= robot_params_.dt_ctrl;
        } else {
            timeout_for_no_feet_in_contact_ = robot_params_.time_no_feet_in_contact;
            state_ = "FOOT_IN_CONTACT";
        }

        if (timeout_for_no_feet_in_contact_ < 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "No feet in contact for too long");
            if (state_ == "FOOT_IN_CONTACT") {
                state_ = "INIT";
                initialization_done_ = false;
                t_init_traj_ = 0;
            }
        }

        if (state_ == "INIT") {
            t_init_traj_ = 0;
            swing_foot_is_left_ = true;
            auto swing_foot_name = l_foot_frame_id_;
            set_position_limits_for_foot_in_optimization(swing_foot_name);
            if (mode_ == "WALK") {
                swing_foot_traj_.set_desired_foot_raise_height(0.1);
            } else {
                swing_foot_traj_.set_desired_foot_raise_height(0.1); // todo fix, this is strange
            }
            state_ = "RAMP_TO_STARTING_POS";
        }

        if (state_ == "RAMP_TO_STARTING_POS") {
            auto T_BLF_to_BF = get_BLF_to_BF();
            auto T_BF_to_BLF = T_BLF_to_BF.inverse();

            Eigen::Vector3d stance_foot_BF = get_eigen_transform(r_foot_frame_id_, base_link_frame_id_).translation();
            stance_foot_BF_saved_ = stance_foot_BF;
            Eigen::Vector3d stance_foot_BLF = T_BF_to_BLF * stance_foot_BF;

            T_STF_to_BLF_.linear() = Eigen::Matrix3d::Identity();
            T_STF_to_BLF_.translation() = stance_foot_BLF;
            broadcast_transform(base_link_frame_id_, "BLF", T_BLF_to_BF.translation(), Eigen::Quaterniond(T_BLF_to_BF.rotation()));
            broadcast_transform("BLF", "STF", T_STF_to_BLF_.translation(), Eigen::Quaterniond(T_STF_to_BLF_.rotation()));

            Eigen::Vector3d fin_swing_foot_pos_STF;
            fin_swing_foot_pos_STF = Eigen::Vector3d(0.0, 0.1, 0.1);

            Eigen::Vector3d fin_baselink_pos_STF;
            fin_baselink_pos_STF = Eigen::Vector3d(0.0, 0.0, robot_params_.robot_height);

            std::string frame_id = "STF";
            publish_body_trajectories(frame_id, fin_baselink_pos_STF, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),  // todo fix this so it takes the init orientation of the robot
                                        Eigen::Vector3d::Zero(), Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                        fin_swing_foot_pos_STF, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

            biped_bringup::msg::StampedBool des_contact_msg;
            des_contact_msg.header.stamp = this->get_clock()->now();
            des_contact_msg.data = false;
            pub_desired_left_contact_->publish(des_contact_msg);
            des_contact_msg.data = false;
            pub_desired_right_contact_->publish(des_contact_msg);

            t_init_traj_ += robot_params_.dt_ctrl;
            if (t_init_traj_ > robot_params_.duration_init_traj) {
                initialization_done_ = true;
                set_starting_to_walk_params(fin_swing_foot_pos_STF);
                t_init_traj_ = robot_params_.duration_init_traj;
            }
        }

        if (state_ == "FOOT_IN_CONTACT" && initialization_done_ == true && mode_ == "WALK") {
            auto t_start = std::chrono::high_resolution_clock::now();
            run_capture_point_controller();
            auto t_finish = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_finish - t_start);

            double dt_ctrl_ms = robot_params_.dt_ctrl * 1000;
            if (duration_ms.count() > dt_ctrl_ms * 0.8)
            {
                RCLCPP_WARN(this->get_logger(), "duration of capture point controller in ms %ld ", duration_ms.count());
                RCLCPP_WARN(this->get_logger(), "dt_ctrl_ms %f", dt_ctrl_ms);
                RCLCPP_WARN(this->get_logger(), "Running slower than desired");
            }
        }

        // if (initialization_done_ == true && mode_ == "ONE_FOOT_SWING") {
        //     run_one_foot_swing();
        // }

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Walking mode: %s", mode_.c_str());
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Walk slow: %d", walk_slow_);



    }

    void run_capture_point_controller()
    {
        bool swing_foot_contact;
        std::string swing_foot_name;
        std::string stance_foot_name;
        if (swing_foot_is_left_) {
            swing_foot_contact = foot_left_contact_;
            swing_foot_name = l_foot_frame_id_;
            stance_foot_name = r_foot_frame_id_;
        } else {
            swing_foot_contact = foot_right_contact_;
            swing_foot_name = r_foot_frame_id_;
            stance_foot_name = l_foot_frame_id_;
        }

        auto T_BLF_to_BF = get_BLF_to_BF();
        broadcast_transform(base_link_frame_id_, "BLF", T_BLF_to_BF.translation(), Eigen::Quaterniond(T_BLF_to_BF.rotation()));
        auto T_BF_to_BLF = T_BLF_to_BF.inverse();

        if (time_since_last_step_ > robot_params_.T_contact_ignore && swing_foot_contact == true) {
            swing_foot_is_left_ = !swing_foot_is_left_;
            std::swap(stance_foot_name, swing_foot_name);

            Eigen::Vector3d stance_foot_BF = get_eigen_transform(stance_foot_name, base_link_frame_id_).translation();
            Eigen::Vector3d swing_foot_BF = get_eigen_transform(swing_foot_name, base_link_frame_id_).translation();

            auto swing_foot_BLF = T_BF_to_BLF * swing_foot_BF;
            Eigen::Vector3d stance_foot_BLF = T_BF_to_BLF * stance_foot_BF;
            T_STF_to_BLF_.linear() = Eigen::Matrix3d::Identity();
            T_STF_to_BLF_.translation() = stance_foot_BLF;
            auto swing_foot_STF = T_STF_to_BLF_.inverse() * swing_foot_BLF;
            swing_foot_position_beginning_of_step_STF_ = swing_foot_STF;
            start_opt_pos_swing_foot_ = swing_foot_position_beginning_of_step_STF_;
            start_opt_vel_swing_foot_ = Eigen::Vector3d::Zero();

            foot_traj_list_STF_.clear(); // used for markers
            swing_foot_traj_.set_initial_pos_vel(start_opt_pos_swing_foot_, start_opt_vel_swing_foot_);
            set_position_limits_for_foot_in_optimization(swing_foot_name);

            if (walk_slow_ == true) {
                swing_foot_traj_.enable_lowering_foot_after_opt_solved(true);
            } else {
                swing_foot_traj_.enable_lowering_foot_after_opt_solved(true);
            }
            time_since_last_step_ = 0.0;
            dcm_at_step_STF_ = dcm_STF_;
        }

        Eigen::Vector3d swing_foot_BF = get_eigen_transform(swing_foot_name, base_link_frame_id_).translation();
        Eigen::Vector3d stance_foot_BF = get_eigen_transform(stance_foot_name, base_link_frame_id_).translation();
        Eigen::Vector3d stance_foot_BLF = T_BF_to_BLF * stance_foot_BF;
        T_STF_to_BLF_.linear() = Eigen::Matrix3d::Identity();
        T_STF_to_BLF_.translation() = stance_foot_BLF; // todo figure out if i update TSTF
        broadcast_transform("BLF", "STF", T_STF_to_BLF_.translation(), Eigen::Quaterniond(T_STF_to_BLF_.rotation()));

        // Desired DCM Trajectory
        Eigen::Vector3d dcm_desired_STF;
        dcm_desired_STF << 1.0/robot_params_.omega*vel_d_[0], 0.0, 0.0;
        if (swing_foot_is_left_) {
            dcm_desired_STF(1) = -0.036 + 1.0/robot_params_.omega * vel_d_[1];
        } else {
            dcm_desired_STF(1) = 0.036 + 1.0/robot_params_.omega * vel_d_[1];
        }

        geometry_msgs::msg::Vector3Stamped dcm_desired_STF_msg;
        get_vector3_msg(dcm_desired_STF, dcm_desired_STF_msg);
        pub_desired_dcm_->publish(dcm_desired_STF_msg);

        Eigen::Vector3d base_link_vel_BF;
        base_link_vel_BF << base_link_odom_.linear_velocity(0), base_link_odom_.linear_velocity(1), base_link_odom_.linear_velocity(2);
        Eigen::Vector3d base_link_vel_BLF = T_BF_to_BLF.rotation() * base_link_vel_BF;
        Eigen::Vector3d vel_base_link_STF = T_STF_to_BLF_.inverse().rotation() * base_link_vel_BLF;

        Eigen::Vector3d offset_com_baselink;
        offset_com_baselink << robot_params_.offset_baselink_cog_x, robot_params_.offset_baselink_cog_y, robot_params_.offset_baselink_cog_z;
        dcm_STF_(0) = T_STF_to_BLF_.inverse().translation()[0] + offset_com_baselink[0] + 1.0 / robot_params_.omega * vel_base_link_STF(0);
        dcm_STF_(1) = T_STF_to_BLF_.inverse().translation()[1] + offset_com_baselink[1] + 1.0 / robot_params_.omega * vel_base_link_STF(1);
        dcm_STF_(2) = 0;

        auto next_dcm_STF_predicted = dcm_STF_ * exp(robot_params_.omega * remaining_time_in_step_);
        Eigen::Vector3d error_dcm_STF = next_dcm_STF_predicted - dcm_desired_STF;
        error_dcm_STF(2) = 0.0;

        Eigen::Vector3d next_footstep_STF;
        next_footstep_STF = -dcm_desired_STF + dcm_STF_ * exp(robot_params_.omega * remaining_time_in_step_);

        Eigen::Vector3d vec_STF_to_next_CP = next_footstep_STF - Eigen::Vector3d(0.0, 0.0, 0.0);
        auto norm_vec_STF_to_next_CP = sqrt(vec_STF_to_next_CP(0) * vec_STF_to_next_CP(0) + vec_STF_to_next_CP(1) * vec_STF_to_next_CP(1));
        Eigen::Vector3d safe_next_footstep_STF = robot_params_.safety_radius_CP / norm_vec_STF_to_next_CP * vec_STF_to_next_CP;

        if (norm_vec_STF_to_next_CP > robot_params_.safety_radius_CP){
            next_footstep_STF = safe_next_footstep_STF;
        }

        geometry_msgs::msg::Vector3Stamped predicted_dcm_STF_msg;
        get_vector3_msg(dcm_at_step_STF_, predicted_dcm_STF_msg);
        pub_predicted_dcm_->publish(predicted_dcm_STF_msg);

        Eigen::Vector3d des_pos_foot_STF;
        des_pos_foot_STF << next_footstep_STF(0), next_footstep_STF(1), 0;

        if (walk_slow_ == true) {
            if (swing_foot_is_left_) {
                des_pos_foot_STF << 0.00, 0.15, 0.0;
            } else {
                des_pos_foot_STF << 0.00, -0.15, 0.0;
            }
        }

        Eigen::Vector3d pos_desired_swing_foot_STF;
        Eigen::Vector3d vel_desired_swing_foot_STF;
        Eigen::Vector3d acc_desired_swing_foot_STF;

        swing_foot_traj_.compute_traj_pos_vel(time_since_last_step_,
                                        des_pos_foot_STF,
                                        pos_desired_swing_foot_STF,
                                        vel_desired_swing_foot_STF,
                                        acc_desired_swing_foot_STF);

        Eigen::Vector3d pos_body_level_STF = T_STF_to_BLF_.inverse().translation();
        pos_body_level_STF(2) = robot_params_.robot_height;
        Eigen::Quaterniond quat_body_level_STF = Eigen::Quaterniond(T_STF_to_BLF_.inverse().rotation());
        Eigen::Vector3d acc_body_level_STF = Eigen::Vector3d::Zero();
        acc_body_level_STF(0) = robot_params_.omega * robot_params_.omega * (pos_body_level_STF(0) + offset_com_baselink(0));
        acc_body_level_STF(1) = robot_params_.omega * robot_params_.omega * (pos_body_level_STF(1) + offset_com_baselink(1));

        if (walk_slow_ == true) {
            pos_body_level_STF << 0.0, 0.0, robot_params_.robot_height;
            quat_body_level_STF = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            acc_body_level_STF = Eigen::Vector3d::Zero();
            vel_base_link_STF = Eigen::Vector3d::Zero();
        }

        double dt = robot_params_.dt_ctrl;
        remaining_time_in_step_ = robot_params_.t_step - time_since_last_step_;
        time_since_last_step_ = time_since_last_step_ + dt;

        Eigen::Vector3d pos_desired_stance_foot_STF = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel_desired_stance_foot_STF = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc_desired_stance_foot_STF = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat_desired_stance_foot_STF = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        Eigen::Quaterniond quat_desired_swing_foot_STF = quat_body_level_STF;

        std::string frame_id = "STF";

        if (swing_foot_name == r_foot_frame_id_) {
            biped_bringup::msg::StampedBool des_contact_msg;
            des_contact_msg.header.stamp = this->get_clock()->now();
            des_contact_msg.data = false;
            pub_desired_right_contact_->publish(des_contact_msg);
            des_contact_msg.data = true;
            pub_desired_left_contact_->publish(des_contact_msg);

            if (error_dcm_STF.squaredNorm() < 0.3)  {
                publish_body_trajectories(frame_id, pos_body_level_STF, quat_body_level_STF, vel_base_link_STF, acc_body_level_STF,
                                                    pos_desired_swing_foot_STF, quat_desired_swing_foot_STF, vel_desired_swing_foot_STF, acc_desired_swing_foot_STF,
                                                    pos_desired_stance_foot_STF, quat_desired_stance_foot_STF, vel_desired_stance_foot_STF, acc_desired_stance_foot_STF);
            }

        } else {
            biped_bringup::msg::StampedBool des_contact_msg;
            des_contact_msg.header.stamp = this->get_clock()->now();
            des_contact_msg.data = false;
            pub_desired_left_contact_->publish(des_contact_msg);
            des_contact_msg.data = true;
            pub_desired_right_contact_->publish(des_contact_msg);

            if (error_dcm_STF.squaredNorm() < 0.3)  {
                publish_body_trajectories(frame_id, pos_body_level_STF, quat_body_level_STF, vel_base_link_STF, acc_body_level_STF,
                                                    pos_desired_stance_foot_STF, quat_desired_stance_foot_STF, vel_desired_stance_foot_STF, acc_desired_stance_foot_STF,
                                                    pos_desired_swing_foot_STF, quat_desired_swing_foot_STF, vel_desired_swing_foot_STF, acc_desired_swing_foot_STF);
            }
        }

        // Marker publishers
        int marker_type;
        marker_type = visualization_msgs::msg::Marker::SPHERE;
        publish_marker(marker_type, next_footstep_STF, "next_footstep", "STF", 1, Eigen::Vector3d(1.0, 0.0, 1.0), pub_marker_next_footstep_);
        // pub safety circle around the STF
        std::list<Eigen::Vector3d> safety_circle_points;
        for (int i = 0; i <= 2 * M_PI / 0.1; i++) {
            Eigen::Vector3d safety_circle_point;
            safety_circle_point << robot_params_.safety_radius_CP * cos(i * 0.1), robot_params_.safety_radius_CP * sin(i * 0.1), 0.0;
            safety_circle_points.push_back(safety_circle_point);
        }
        publish_line_traj_markers(safety_circle_points, "safety_circle", "STF", 4, Eigen::Vector3d(0.0, 1.0, 0.0), pub_markers_safety_circle_);

        publish_marker(marker_type, swing_foot_BF, "swing_foot", base_link_frame_id_, 5, Eigen::Vector3d(1.0, 1.0, 0.0), pub_marker_swing_foot_BF_);
        publish_marker(marker_type, stance_foot_BF, "stance_foot", base_link_frame_id_, 6, Eigen::Vector3d(1.0, 1.0, 0.0), pub_marker_stance_foot_BF_);
        foot_traj_list_STF_.push_back(pos_desired_swing_foot_STF);
        publish_line_traj_markers(foot_traj_list_STF_, "foot_trajectory", "STF", 3, Eigen::Vector3d(1.0, 0.0, 1.0), pub_markers_foot_traj_);
    }

    void contact_right_callback(biped_bringup::msg::StampedBool::SharedPtr msg)
    {
        foot_right_contact_ = msg->data;
    }

    void contact_left_callback(biped_bringup::msg::StampedBool::SharedPtr msg)
    {
        foot_left_contact_ = msg->data;
    }

    void vel_cmd_cb(geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {
        vel_d_[0] = msg->twist.linear.x;
        vel_d_[1] = msg->twist.linear.y;
        vel_d_[2] = msg->twist.angular.z;
    }

    Eigen::Transform<double, 3, Eigen::AffineCompact> get_BLF_to_BF()
    {
        Eigen::Quaterniond quat_BF_to_IF = base_link_odom_.orientation;

        Eigen::Vector3d x_vec_BF;
        x_vec_BF << 1.0, 0.0, 0.0;
        Eigen::Vector3d x_vec_IF = quat_BF_to_IF * x_vec_BF;
        Eigen::Vector3d bl_x_vec_IF(x_vec_IF(0), x_vec_IF(1), 0.0);
        bl_x_vec_IF.normalize();
        Eigen::Vector3d bl_z_vec_IF(0.0, 0.0, 1.0);
        Eigen::Vector3d bl_y_vec_IF = bl_z_vec_IF.cross(bl_x_vec_IF);

        Eigen::Matrix3d R_BLF_to_IF;
        R_BLF_to_IF << bl_x_vec_IF, bl_y_vec_IF, bl_z_vec_IF;

        Eigen::Transform<double, 3, Eigen::AffineCompact> T_BLF_to_BF;
        T_BLF_to_BF.linear() = quat_BF_to_IF.conjugate().toRotationMatrix() * R_BLF_to_IF;
        T_BLF_to_BF.translation() = Eigen::Vector3d::Zero();
        return T_BLF_to_BF;
    }

    geometry_msgs::msg::TransformStamped get_transform(std::string fromFrameRel, std::string toFrameRel)
    {
        geometry_msgs::msg::TransformStamped t;
        try
        {
            t = tf_buffer_->lookupTransform(
                toFrameRel, fromFrameRel,
                tf2::TimePointZero, tf2::durationFromSec(0.0));

            auto now = this->get_clock()->now();
            double dt = (now - rclcpp::Time(t.header.stamp)).seconds();
            if (dt > 0.1) {
                RCLCPP_WARN(this->get_logger(), "TF is too old: %f", dt);
            }
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not transform %s to %s: %s",
                         fromFrameRel.c_str(), toFrameRel.c_str(), ex.what());
            throw std::runtime_error("could not complete the transform");
        }

        return t;
    }

    Eigen::Transform<double, 3, Eigen::AffineCompact> get_eigen_transform(std::string fromFrameRel, std::string toFrameRel)
    {
        auto t = get_transform(fromFrameRel, toFrameRel);

        Eigen::Quaterniond rot;
        rot.x() = t.transform.rotation.x;
        rot.y() = t.transform.rotation.y;
        rot.z() = t.transform.rotation.z;
        rot.w() = t.transform.rotation.w;
        Eigen::Vector3d pos;
        pos(0) = t.transform.translation.x;
        pos(1) = t.transform.translation.y;
        pos(2) = t.transform.translation.z;

        Eigen::Transform<double, 3, Eigen::AffineCompact> transform;
        transform.fromPositionOrientationScale(pos, rot, Eigen::Vector3d(1.0, 1.0, 1.0));
        return transform;
    }

    void broadcast_transform(std::string frame_id, std::string child_frame_id, Eigen::Vector3d pt_pos, Eigen::Quaterniond pt_quat)
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->get_clock()->now();
        t.header.frame_id = frame_id;
        t.child_frame_id = child_frame_id;
        t.transform.translation.x = pt_pos(0);
        t.transform.translation.y = pt_pos(1);
        t.transform.translation.z = pt_pos(2);
        t.transform.rotation.x = pt_quat.x();
        t.transform.rotation.y = pt_quat.y();
        t.transform.rotation.z = pt_quat.z();
        t.transform.rotation.w = pt_quat.w();
        tf_broadcaster_->sendTransform(t);
    }

    void publish_marker(int marker_type, Eigen::Vector3d pos, std::string name_marker, std::string frame_id, int id, Eigen::Vector3d color, rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub)
    {
        visualization_msgs::msg::Marker marker_msg;
        marker_msg.header.frame_id = frame_id;
        marker_msg.header.stamp = this->get_clock()->now();
        marker_msg.ns = name_marker;
        marker_msg.id = id;
        marker_msg.type = marker_type;
        marker_msg.action = visualization_msgs::msg::Marker::ADD;
        marker_msg.pose.position.x = pos[0];
        marker_msg.pose.position.y = pos[1];
        marker_msg.pose.position.z = pos[2];
        marker_msg.pose.orientation.x = 0;
        marker_msg.pose.orientation.y = 0;
        marker_msg.pose.orientation.z = 0;
        marker_msg.pose.orientation.w = 1;
        marker_msg.scale.x = 0.05;
        marker_msg.scale.y = 0.05;
        marker_msg.scale.z = 0.05;
        marker_msg.color.a = 1.0;
        marker_msg.color.r = color(0);
        marker_msg.color.g = color(1);
        marker_msg.color.b = color(2);
        pub->publish(marker_msg);
    }

    void publish_arrow_marker(Eigen::Vector3d pos_start, Eigen::Vector3d pos_end, std::string name_marker, std::string frame_id, int id, Eigen::Vector3d color, rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub)
    {
        visualization_msgs::msg::Marker marker_msg;
        marker_msg.header.frame_id = frame_id;
        marker_msg.header.stamp = this->get_clock()->now();
        marker_msg.ns = name_marker;
        marker_msg.id = id;
        marker_msg.type = visualization_msgs::msg::Marker::ARROW;
        marker_msg.action = visualization_msgs::msg::Marker::ADD;

        geometry_msgs::msg::Point p;
        p.x = pos_start(0);
        p.y = pos_start(1);
        p.z = pos_start(2);
        marker_msg.points.push_back(p);
        p.x = pos_end(0);
        p.y = pos_end(1);
        p.z = pos_end(2);
        marker_msg.points.push_back(p);
        marker_msg.pose.orientation.w = 1;
        marker_msg.scale.x = 0.05;
        marker_msg.scale.y = 0.05;
        marker_msg.scale.z = 0.05;
        marker_msg.color.a = 1.0;
        marker_msg.color.r = color(0);
        marker_msg.color.g = color(1);
        marker_msg.color.b = color(2);
        pub->publish(marker_msg);
    }

    void publish_line_traj_markers(std::list<Eigen::Vector3d> pos_list, std::string name_marker, std::string frame_id, int id, Eigen::Vector3d color, rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub)
    {
        visualization_msgs::msg::Marker marker_msg;
        marker_msg.header.frame_id = frame_id;
        marker_msg.header.stamp = this->get_clock()->now();
        marker_msg.ns = name_marker;
        marker_msg.id = id;
        marker_msg.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker_msg.action = visualization_msgs::msg::Marker::ADD;
        marker_msg.lifetime = rclcpp::Duration(10, 0);

        marker_msg.pose.orientation.w = 1;
        marker_msg.scale.x = 0.01;
        marker_msg.color.a = 1.0;
        marker_msg.color.r = color(0);
        marker_msg.color.g = color(1);
        marker_msg.color.b = color(2);

        for (auto pos : pos_list)
        {
            geometry_msgs::msg::Point p;
            p.x = pos[0];
            p.y = pos[1];
            p.z = pos[2];
            marker_msg.points.push_back(p);
        }

        pub->publish(marker_msg);
    }

    void publish_body_trajectories(std::string frame_id, Eigen::Vector3d pos_body, Eigen::Quaterniond quat_body, Eigen::Vector3d vel_body, Eigen::Vector3d acc_body,
                                   Eigen::Vector3d pos_right_foot, Eigen::Quaterniond quat_right_foot, Eigen::Vector3d vel_right_foot, Eigen::Vector3d acc_right_foot,
                                   Eigen::Vector3d pos_left_foot, Eigen::Quaterniond quat_left_foot, Eigen::Vector3d vel_left_foot, Eigen::Vector3d acc_left_foot)
    {
        trajectory_msgs::msg::MultiDOFJointTrajectory msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id;
        msg.joint_names.push_back(base_link_frame_id_);
        msg.joint_names.push_back(r_foot_urdf_frame_id_);
        msg.joint_names.push_back(l_foot_urdf_frame_id_);

        geometry_msgs::msg::Transform body;
        body.translation.x = pos_body(0);
        body.translation.y = pos_body(1);
        body.translation.z = pos_body(2);
        body.rotation.x = quat_body.x();
        body.rotation.y = quat_body.y();
        body.rotation.z = quat_body.z();
        body.rotation.w = quat_body.w();

        geometry_msgs::msg::Twist body_vel;
        body_vel.linear.x = vel_body(0);
        body_vel.linear.y = vel_body(1);
        body_vel.linear.z = vel_body(2);
        geometry_msgs::msg::Twist body_acc;
        body_acc.linear.x = acc_body(0);
        body_acc.linear.y = acc_body(1);
        body_acc.linear.z = acc_body(2);

        geometry_msgs::msg::Transform right_foot;
        right_foot.translation.x = pos_right_foot(0);
        right_foot.translation.y = pos_right_foot(1);
        right_foot.translation.z = pos_right_foot(2);
        right_foot.rotation.x = quat_right_foot.x();
        right_foot.rotation.y = quat_right_foot.y();
        right_foot.rotation.z = quat_right_foot.z();
        right_foot.rotation.w = quat_right_foot.w();

        geometry_msgs::msg::Twist right_foot_vel;
        right_foot_vel.linear.x = vel_right_foot(0);
        right_foot_vel.linear.y = vel_right_foot(1);
        right_foot_vel.linear.z = vel_right_foot(2);
        geometry_msgs::msg::Twist right_foot_acc;
        right_foot_acc.linear.x = acc_right_foot(0);
        right_foot_acc.linear.y = acc_right_foot(1);
        right_foot_acc.linear.z = acc_right_foot(2);

        geometry_msgs::msg::Transform left_foot;
        left_foot.translation.x = pos_left_foot(0);
        left_foot.translation.y = pos_left_foot(1);
        left_foot.translation.z = pos_left_foot(2);
        left_foot.rotation.x = quat_left_foot.x();
        left_foot.rotation.y = quat_left_foot.y();
        left_foot.rotation.z = quat_left_foot.z();
        left_foot.rotation.w = quat_left_foot.w();

        geometry_msgs::msg::Twist left_foot_vel;
        left_foot_vel.linear.x = vel_left_foot(0);
        left_foot_vel.linear.y = vel_left_foot(1);
        left_foot_vel.linear.z = vel_left_foot(2);
        geometry_msgs::msg::Twist left_foot_acc;
        left_foot_acc.linear.x = acc_left_foot(0);
        left_foot_acc.linear.y = acc_left_foot(1);
        left_foot_acc.linear.z = acc_left_foot(2);

        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint body_traj_point;
        body_traj_point.transforms.push_back(body);
        body_traj_point.velocities.push_back(body_vel);
        body_traj_point.accelerations.push_back(body_acc);

        body_traj_point.transforms.push_back(right_foot);
        body_traj_point.velocities.push_back(right_foot_vel);
        body_traj_point.accelerations.push_back(right_foot_acc);
        body_traj_point.transforms.push_back(left_foot);
        body_traj_point.velocities.push_back(left_foot_vel);
        body_traj_point.accelerations.push_back(left_foot_acc);

        msg.points.push_back(body_traj_point);
        pub_body_trajectory_->publish(msg);
    }

    void get_vector3_msg(Eigen::Vector3d vec, geometry_msgs::msg::Vector3Stamped &msg)
    {
        msg.header.stamp = this->now();
        msg.vector.x = vec(0);
        msg.vector.y = vec(1);
        msg.vector.z = vec(2);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_body_trajectory_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_next_footstep_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_dcm_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_desired_dcm_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_stance_foot_BF_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_swing_foot_BF_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_markers_foot_traj_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_markers_foot_traj_actual_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_markers_safety_circle_;

    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_desired_dcm_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_predicted_dcm_;

    rclcpp::Publisher<biped_bringup::msg::StampedBool>::SharedPtr pub_desired_left_contact_;
    rclcpp::Publisher<biped_bringup::msg::StampedBool>::SharedPtr pub_desired_right_contact_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr contact_right_sub_;
    rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr contact_left_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_cmd_sub_;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr e_stop_sub_;
    bool e_stop_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string r_foot_frame_id_;
    std::string l_foot_frame_id_;
    std::string r_foot_urdf_frame_id_;
    std::string l_foot_urdf_frame_id_;
    std::string base_link_frame_id_;

    Eigen::Transform<double, 3, Eigen::AffineCompact> T_STF_to_BLF_;

    struct
    {
        double robot_height;
        double t_step;
        double omega;
        double dt_ctrl;
        double duration_init_traj;
        double safety_radius_CP;
        double T_contact_ignore;
        double offset_baselink_cog_x;
        double offset_baselink_cog_y;
        double offset_baselink_cog_z;
        double time_no_feet_in_contact;
        double foot_separation;
        double swing_x_safe_box_min;
        double swing_x_safe_box_max;
        double swing_y_safe_box_min;
        double swing_y_safe_box_max;
        double swing_z_safe_box_min;
        double swing_z_safe_box_max;
        bool walk_slow;
    } robot_params_;

    bool foot_right_contact_ = false;
    bool foot_left_contact_ = false;

    struct
    {
        rclcpp::Time stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d linear_velocity;
        Eigen::Vector3d angular_velocity;
        Eigen::Vector3d linear_acceleration;
        Eigen::Vector3d angular_acceleration;
    } base_link_odom_;

    bool swing_foot_is_left_;

    Eigen::Vector3d swing_foot_position_beginning_of_step_STF_;
    std::list<Eigen::Vector3d> foot_traj_list_STF_;
    std::list<Eigen::Vector3d> foot_actual_traj_list_STF_;
    float time_since_last_step_;
    float remaining_time_in_step_;

    Eigen::Vector3d vel_d_;
    Eigen::Vector3d dcm_STF_;
    Eigen::Vector3d dcm_at_step_STF_;

    std::string state_;
    std::string mode_;
    int sign_ = 1;

    double timeout_for_no_feet_in_contact_ = 0.0;

    double t_init_traj_;
    bool initialization_done_;

    OptimizerTrajectory swing_foot_traj_;
    Eigen::Vector3d start_opt_pos_swing_foot_, start_opt_vel_swing_foot_;
    Eigen::Vector3d stance_foot_BF_saved_;

    bool start_cmd_line_;
    bool walk_slow_;


};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CapturePoint>());
    rclcpp::shutdown();
    return 0;
}