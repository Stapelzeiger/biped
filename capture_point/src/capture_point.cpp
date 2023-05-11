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

#include "foot_trajectory.h"

using namespace std::placeholders;
using namespace std::chrono_literals;

class CapturePoint : public rclcpp::Node
{

public:
    CapturePoint() : Node("cp_node")
    {
        robot_params.robot_height = this->declare_parameter<double>("robot_height",  0.54);
        robot_params.t_step = this->declare_parameter<double>("t_step", 0.25);
        robot_params.dt_ctrl = this->declare_parameter<double>("ctrl_time_sec", 0.01);
        robot_params.duration_init_traj = this->declare_parameter<double>("duration_init_traj", 3.0);
        robot_params.safety_radius_CP = this->declare_parameter<double>("safety_radius_CP", 1.0);
        robot_params.T_contact_ignore = this->declare_parameter<double>("T_contact_ignore", 0.1);
        robot_params.omega = sqrt(9.81 / robot_params.robot_height);
        counter_calib_traj_ = 0;

        state_ = "INIT";
        initialization_done_ = false;
        t_init_traj_ = 0.0;
        foot_trajectory_ = FootTrajectory(robot_params.t_step, robot_params.dt_ctrl);

        r_foot_frame_id_ = this->declare_parameter<std::string>("r_foot_frame_id", "R_FOOT");
        l_foot_frame_id_ = this->declare_parameter<std::string>("l_foot_frame_id", "L_FOOT");
        r_foot_urdf_frame_id_ = this->declare_parameter<std::string>("r_foot_urdf_frame_id", "R_FOOT");
        l_foot_urdf_frame_id_ = this->declare_parameter<std::string>("l_foot_urdf_frame_id", "L_FOOT");
        base_link_frame_id_ = this->declare_parameter<std::string>("base_link_frame_id", "base_link");

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        pub_body_trajectory_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/body_trajectories", 10);

        pub_markers_foot_traj_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_traj_feet", 10);
        pub_markers_safety_circle_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_safety_circle", 10);
        pub_marker_next_footstep_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_next_footstep", 10);
        pub_marker_next_safe_footstep_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_next_safe_footstep", 10);
        pub_marker_dcm_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_dcm", 10);
        pub_marker_desired_dcm_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_desired_dcm", 10);
        pub_marker_stance_foot_BF_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_stance_foot_BF", 10);
        pub_marker_swing_foot_BF_ = this->create_publisher<visualization_msgs::msg::Marker>("~/markers_swing_foot_BF", 10);
        pub_marker_vel_BF_ = this->create_publisher<visualization_msgs::msg::Marker>("~/marker_vel_BF", 10);

        pub_desired_left_contact_ = this->create_publisher<biped_bringup::msg::StampedBool>("~/desired_left_contact", 10);
        pub_desired_right_contact_ = this->create_publisher<biped_bringup::msg::StampedBool>("~/desired_right_contact", 10);

        pub_swing_foot_BF_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("~/swing_foot_BF", 10);
        pub_stance_foot_BF_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("~/stance_foot_BF", 10);
        pub_dcm_desired_BF_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("~/dcm_desired_BF", 10);

        pub_state_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("~/state", 10);

        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", 10, std::bind(&CapturePoint::odometry_callback, this, _1));

        contact_right_sub_ = this->create_subscription<biped_bringup::msg::StampedBool>(
            "~/contact_foot_right", 10, std::bind(&CapturePoint::contact_right_callback, this, _1));

        contact_left_sub_ = this->create_subscription<biped_bringup::msg::StampedBool>(
            "~/contact_foot_left", 10, std::bind(&CapturePoint::contact_left_callback, this, _1));

        vel_cmd_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            "~/vel_cmd", 10, std::bind(&CapturePoint::vel_cmd_cb, this, _1));

        std::chrono::duration<double> period = robot_params.dt_ctrl * 1s;
        timer_ = rclcpp::create_timer(this, this->get_clock(), period, std::bind(&CapturePoint::timer_callback, this));
    }

private:
    void set_starting_to_walk_params(Eigen::Vector3d swing_foot_STF)
    {
        foot_right_contact_ = false;
        foot_left_contact_ = false;

        time_since_last_step_ = robot_params.t_step / 2;
        remaining_time_in_step_ = robot_params.t_step - time_since_last_step_;
        timeout_for_no_feet_in_contact_ = 0;

        swing_foot_position_beginning_of_step_STF_ = swing_foot_STF;
        previous_base_link_vel_BF_ << 0.0, 0.0, 0.0;
        foot_traj_list_STF_.clear();
        foot_trajectory_.set_initial_position(swing_foot_position_beginning_of_step_STF_);
    }

    void get_foot_setpt(std::vector<Eigen::Vector<double, 4>> coeffs, double t, Eigen::Vector3d &foot_pos, Eigen::Vector3d &foot_vel)
    {
        auto coef_x = coeffs[0];
        auto coef_y = coeffs[1];
        auto coef_z = coeffs[2];

        foot_pos[0] = foot_trajectory_.get_q(coef_x, t);
        foot_pos[1] = foot_trajectory_.get_q(coef_y, t);
        foot_pos[2] = foot_trajectory_.get_q(coef_z, t);

        foot_vel[0] = foot_trajectory_.get_q_dot(coef_x, t);
        foot_vel[1] = foot_trajectory_.get_q_dot(coef_y, t);
        foot_vel[2] = foot_trajectory_.get_q_dot(coef_z, t);
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
        if (base_link_odom_.stamp == rclcpp::Time(0, 0, RCL_ROS_TIME))
        {
            RCLCPP_INFO(this->get_logger(), "Waiting for odometry...");
            return;
        }
        if ((this->get_clock()->now() - base_link_odom_.stamp).seconds() > 0.1)
        {
            RCLCPP_ERROR(this->get_logger(), "Odometry is too old");
            return;
        }

        if (!tf_buffer_->canTransform(base_link_frame_id_, r_foot_frame_id_, tf2::TimePointZero) ||
            !tf_buffer_->canTransform(base_link_frame_id_, l_foot_frame_id_, tf2::TimePointZero))
        {
            RCLCPP_INFO(this->get_logger(), "Waiting for TFs for R_FOOT and L_FOOT...");
            return;
        }

        if (foot_left_contact_ == false && foot_right_contact_ == false)
        {
            timeout_for_no_feet_in_contact_ -= robot_params.dt_ctrl;
            std::cout << "lost contact of the feet" << std::endl;
        }
        else
        {
            timeout_for_no_feet_in_contact_ = 0.2;
            state_ = "FOOT_IN_CONTACT";
        }

        if (timeout_for_no_feet_in_contact_ < 0)
        {
            if (state_ == "FOOT_IN_CONTACT")
            {
                state_ = "INIT";
                initialization_done_ = false;
                t_init_traj_ = 0;
            }
        }

        geometry_msgs::msg::Vector3Stamped state_msg;
        state_msg.header.stamp = this->get_clock()->now();
        if (state_ == "INIT")
            state_msg.vector.x = 0;
        else if (state_ == "RAMP_TO_STARTING_POS")
            state_msg.vector.x = 1;
        else if (state_ == "FOOT_IN_CONTACT")
            state_msg.vector.x = 2;
        pub_state_->publish(state_msg);

        if (state_ == "INIT")
        {
            swing_foot_is_left_ = true;

            // setup frames:
            auto T_BLF_to_BF = get_BLF_to_BF();
            auto T_BF_to_BLF = T_BLF_to_BF.inverse();

            Eigen::Vector3d stance_foot_BF = get_eigen_transform(r_foot_frame_id_, base_link_frame_id_).translation();
            Eigen::Vector3d swing_foot_BF = get_eigen_transform(l_foot_frame_id_, base_link_frame_id_).translation();
            auto swing_foot_BLF = T_BF_to_BLF * swing_foot_BF;

            Eigen::Vector3d stance_foot_BLF = T_BF_to_BLF * stance_foot_BF;
            T_STF_to_BLF_.linear() = Eigen::Matrix3d::Identity();
            T_STF_to_BLF_.translation() = stance_foot_BLF;
            broadcast_transform(base_link_frame_id_, "BLF", T_BLF_to_BF.translation(), Eigen::Quaterniond(T_BLF_to_BF.rotation()));
            broadcast_transform("BLF", "STF", T_STF_to_BLF_.translation(), Eigen::Quaterniond(T_STF_to_BLF_.rotation()));

            Eigen::Vector3d init_swing_foot_pos_STF, fin_swing_foot_pos_STF;
            init_swing_foot_pos_STF = T_STF_to_BLF_.inverse()*swing_foot_BLF;
            fin_swing_foot_pos_STF = Eigen::Vector3d(0.0, 0.2, 0.10);

            Eigen::Vector3d init_baselink_pos_STF, fin_baselink_pos_STF;
            init_baselink_pos_STF = T_STF_to_BLF_.inverse().translation();
            fin_baselink_pos_STF = Eigen::Vector3d(0.0, -0.1, robot_params.robot_height);

            for (int i = 0; i < 3; i++)
            {
                auto coeff = foot_trajectory_.get_spline_coef(robot_params.duration_init_traj,
                                                            init_swing_foot_pos_STF(i), 0.0,
                                                            fin_swing_foot_pos_STF(i), 0.0);
                coeffs_swing_foot_init_traj_.push_back(coeff);

                coeff = foot_trajectory_.get_spline_coef(robot_params.duration_init_traj,
                                                            init_baselink_pos_STF(i), 0.0,
                                                            fin_baselink_pos_STF(i), 0.0);
                coeffs_baselink_init_traj_.push_back(coeff);
            }
            state_ = "RAMP_TO_STARTING_POS";
            foot_trajectory_.set_initial_position(init_swing_foot_pos_STF);
            previous_vel_base_link_STF_.setZero();
        }

        if (state_ == "RAMP_TO_STARTING_POS")
        {
            // setup frames:
            auto T_BLF_to_BF = get_BLF_to_BF();
            auto T_BF_to_BLF = T_BLF_to_BF.inverse();

            Eigen::Vector3d stance_foot_BF = get_eigen_transform(r_foot_frame_id_, base_link_frame_id_).translation();
            Eigen::Vector3d swing_foot_BF = get_eigen_transform(l_foot_frame_id_, base_link_frame_id_).translation();
            auto swing_foot_BLF = T_BF_to_BLF * swing_foot_BF;

            Eigen::Vector3d stance_foot_BLF = T_BF_to_BLF * stance_foot_BF;
            T_STF_to_BLF_.linear() = Eigen::Matrix3d::Identity();
            T_STF_to_BLF_.translation() = stance_foot_BLF;
            broadcast_transform(base_link_frame_id_, "BLF", T_BLF_to_BF.translation(), Eigen::Quaterniond(T_BLF_to_BF.rotation()));
            broadcast_transform("BLF", "STF", T_STF_to_BLF_.translation(), Eigen::Quaterniond(T_STF_to_BLF_.rotation()));

            Eigen::Vector3d setpt_baselink_foot_pos_STF, setpt_swing_foot_pos_STF;
            Eigen::Vector3d setpt_baselink_foot_vel_STF, setpt_swing_foot_vel_STF;

            get_foot_setpt(coeffs_baselink_init_traj_, t_init_traj_, setpt_baselink_foot_pos_STF, setpt_baselink_foot_vel_STF);
            get_foot_setpt(coeffs_swing_foot_init_traj_, t_init_traj_, setpt_swing_foot_pos_STF, setpt_swing_foot_vel_STF);

            std::string frame_id = "STF";
            publish_body_trajectories(frame_id, setpt_baselink_foot_pos_STF, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),  // todo fix this so it takes the init orientation of the robot
                                        Eigen::Vector3d::Zero(), Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                        setpt_swing_foot_pos_STF, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

            biped_bringup::msg::StampedBool des_contact_msg;
            des_contact_msg.header.stamp = this->get_clock()->now();
            des_contact_msg.data = false;
            pub_desired_left_contact_->publish(des_contact_msg);
            des_contact_msg.data = false;
            pub_desired_right_contact_->publish(des_contact_msg);

            t_init_traj_ += robot_params.dt_ctrl;
            if (t_init_traj_ > robot_params.duration_init_traj)
            {
                initialization_done_ = true;
                set_starting_to_walk_params(setpt_swing_foot_pos_STF);
                t_init_traj_ = robot_params.duration_init_traj;
                std::cout << "Starting to walk! RESET!" << std::endl;
            }
        }
        if (state_ == "FOOT_IN_CONTACT" && initialization_done_ == true)
        {
            run_capture_point_controller();
        }
    }

    void run_capture_point_controller()
    {
        bool swing_foot_contact;
        std::string swing_foot_name;
        std::string stance_foot_name;
        if (swing_foot_is_left_)
        {
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
        if (time_since_last_step_ > robot_params.T_contact_ignore && swing_foot_contact == true)
        {
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
            foot_trajectory_.set_initial_position(swing_foot_position_beginning_of_step_STF_);
            foot_traj_list_STF_.clear(); // used for markers

            time_since_last_step_ = 0.0;
        }

        Eigen::Vector3d swing_foot_BF = get_eigen_transform(swing_foot_name, base_link_frame_id_).translation();
        Eigen::Vector3d stance_foot_BF = get_eigen_transform(stance_foot_name, base_link_frame_id_).translation();

        geometry_msgs::msg::Vector3Stamped swing_foot_msg;
        swing_foot_msg.header.stamp = this->get_clock()->now();
        swing_foot_msg.vector.x = swing_foot_BF(0);
        swing_foot_msg.vector.y = swing_foot_BF(1);
        swing_foot_msg.vector.z = swing_foot_BF(2);
        pub_swing_foot_BF_->publish(swing_foot_msg);

        geometry_msgs::msg::Vector3Stamped stance_foot_msg;
        stance_foot_msg.header.stamp = this->get_clock()->now();
        stance_foot_msg.vector.x = stance_foot_BF(0);
        stance_foot_msg.vector.y = stance_foot_BF(1);
        stance_foot_msg.vector.z = stance_foot_BF(2);
        pub_stance_foot_BF_->publish(stance_foot_msg);

        Eigen::Vector3d stance_foot_BLF = T_BF_to_BLF * stance_foot_BF;
        T_STF_to_BLF_.linear() = Eigen::Matrix3d::Identity();
        T_STF_to_BLF_.translation() = stance_foot_BLF; // todo figure out if i update TSTF
        broadcast_transform("BLF", "STF", T_STF_to_BLF_.translation(), Eigen::Quaterniond(T_STF_to_BLF_.rotation()));

        // Desired DCM Trajectory
        Eigen::Vector3d dcm_desired_STF;
        dcm_desired_STF << vel_d_[0], 0.0, 0.0;
        if (swing_foot_is_left_)
        {
            dcm_desired_STF(1) = -0.036 + vel_d_[1];
        }else{
            dcm_desired_STF(1) = 0.036 + vel_d_[1];
        }
        Eigen::Vector3d dcm_desired_BLF = T_STF_to_BLF_.rotation() * dcm_desired_STF;
        Eigen::Vector3d dcm_desired_BF = T_BLF_to_BF.rotation() * dcm_desired_BLF;
        geometry_msgs::msg::TwistStamped dcm_desired_BF_msg;
        dcm_desired_BF_msg.header.stamp = this->get_clock()->now();
        dcm_desired_BF_msg.twist.linear.x = dcm_desired_BF(0);
        dcm_desired_BF_msg.twist.linear.y = dcm_desired_BF(1);
        dcm_desired_BF_msg.twist.linear.z = dcm_desired_BF(2);
        pub_dcm_desired_BF_->publish(dcm_desired_BF_msg);

        Eigen::Vector3d base_link_vel_BF;
        base_link_vel_BF << base_link_odom_.linear_velocity(0), base_link_odom_.linear_velocity(1), base_link_odom_.linear_velocity(2);
        Eigen::Vector3d base_link_vel_BLF = T_BF_to_BLF.rotation() * base_link_vel_BF;
        Eigen::Vector3d vel_base_link_STF = T_STF_to_BLF_.inverse().rotation() * base_link_vel_BLF;

        Eigen::Vector3d dcm_STF;
        dcm_STF(0) = T_STF_to_BLF_.inverse().translation()[0] + 1.0 / robot_params.omega * vel_base_link_STF(0);
        dcm_STF(1) = T_STF_to_BLF_.inverse().translation()[1] + 1.0 / robot_params.omega * vel_base_link_STF(1);
        dcm_STF(2) = 0;

        Eigen::Vector3d next_footstep_STF;
        next_footstep_STF = -dcm_desired_STF + dcm_STF * exp(robot_params.omega * remaining_time_in_step_);
        // // safety circle for the CP
        // Eigen::Vector3d vec_STF_to_next_CP = next_footstep_STF - Eigen::Vector3d(0.0, 0.0, 0.0);
        // auto norm_vec_STF_to_next_CP = sqrt(vec_STF_to_next_CP(0) * vec_STF_to_next_CP(0) + vec_STF_to_next_CP(1) * vec_STF_to_next_CP(1));
        // Eigen::Vector3d safe_next_footstep_STF = robot_params.safety_radius_CP / norm_vec_STF_to_next_CP * vec_STF_to_next_CP;
        // if (norm_vec_STF_to_next_CP > robot_params.safety_radius_CP)
        // {
        //     next_footstep_STF = safe_next_footstep_STF;
        // }
        Eigen::Vector3d des_pos_foot_STF;
        des_pos_foot_STF << next_footstep_STF(0), next_footstep_STF(1), 0;
        foot_trajectory_.set_desired_end_position(des_pos_foot_STF);

        Eigen::Vector3d pos_desired_swing_foot_STF;
        Eigen::Vector3d vel_desired_swing_foot_STF;
        Eigen::Vector3d acc_desired_swing_foot_STF;

        foot_trajectory_.get_traj_foot_pos_vel(time_since_last_step_, pos_desired_swing_foot_STF, vel_desired_swing_foot_STF, acc_desired_swing_foot_STF);

        double dt = robot_params.dt_ctrl;
        remaining_time_in_step_ = robot_params.t_step - time_since_last_step_;
        time_since_last_step_ = time_since_last_step_ + dt;


        Eigen::Vector3d pos_body_level_STF = T_STF_to_BLF_.inverse().translation();
        pos_body_level_STF(2) = robot_params.robot_height;
        Eigen::Quaterniond quat_body_level_STF = Eigen::Quaterniond(T_STF_to_BLF_.inverse().rotation());
        // Eigen::Vector3d acc_body_level_STF =  (vel_base_link_STF - previous_vel_base_link_STF_)/robot_params.dt_ctrl;
        Eigen::Vector3d acc_body_level_STF = Eigen::Vector3d::Zero();
        // previous_vel_base_link_STF_ = vel_base_link_STF;
        // std::cout << "acc_body_level_STF" << acc_body_level_STF.transpose() << std::endl;

        Eigen::Vector3d pos_desired_stance_foot_STF = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel_desired_stance_foot_STF = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc_desired_stance_foot_STF = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat_desired_stance_foot_STF = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        Eigen::Quaterniond quat_desired_swing_foot_STF = quat_body_level_STF;

        std::string frame_id = "STF";
        const double foot_separation = 0.04;
        Eigen::Vector2d swing_x_safe_box;
        Eigen::Vector2d swing_y_safe_box;
        Eigen::Vector2d swing_z_safe_box;

        if (swing_foot_name == r_foot_frame_id_)
        {
            biped_bringup::msg::StampedBool des_contact_msg;
            des_contact_msg.header.stamp = this->get_clock()->now();
            des_contact_msg.data = false;
            pub_desired_right_contact_->publish(des_contact_msg);
            des_contact_msg.data = true;
            pub_desired_left_contact_->publish(des_contact_msg);

            swing_x_safe_box << -0.2, 0.2;
            swing_y_safe_box << -0.4, -foot_separation*0.5;
            swing_z_safe_box << -0.1, 0.2;

            pos_desired_swing_foot_STF(0) = fmax( fmin( pos_desired_swing_foot_STF(0), swing_x_safe_box(1)), swing_x_safe_box(0));
            pos_desired_swing_foot_STF(1) = fmax( fmin( pos_desired_swing_foot_STF(1), swing_y_safe_box(1)), swing_y_safe_box(0));
            pos_desired_swing_foot_STF(2) = fmax( fmin( pos_desired_swing_foot_STF(2), swing_z_safe_box(1)), swing_z_safe_box(0));


            publish_body_trajectories(frame_id, pos_body_level_STF, quat_body_level_STF, vel_base_link_STF, acc_body_level_STF,
                                                pos_desired_swing_foot_STF, quat_desired_swing_foot_STF, vel_desired_swing_foot_STF, acc_desired_swing_foot_STF,
                                                pos_desired_stance_foot_STF, quat_desired_stance_foot_STF, vel_desired_stance_foot_STF, acc_desired_stance_foot_STF);

        } else {
            biped_bringup::msg::StampedBool des_contact_msg;
            des_contact_msg.header.stamp = this->get_clock()->now();
            des_contact_msg.data = false;
            pub_desired_left_contact_->publish(des_contact_msg);
            des_contact_msg.data = true;
            pub_desired_right_contact_->publish(des_contact_msg);

            swing_x_safe_box << -0.2, 0.2;
            swing_y_safe_box << foot_separation*0.5, 0.4;
            swing_z_safe_box << -0.1, 0.2;

            pos_desired_swing_foot_STF(0) = fmax( fmin( pos_desired_swing_foot_STF(0), swing_x_safe_box(1)), swing_x_safe_box(0));
            pos_desired_swing_foot_STF(1) = fmax( fmin( pos_desired_swing_foot_STF(1), swing_y_safe_box(1)), swing_y_safe_box(0));
            pos_desired_swing_foot_STF(2) = fmax( fmin( pos_desired_swing_foot_STF(2), swing_z_safe_box(1)), swing_z_safe_box(0));

            publish_body_trajectories(frame_id, pos_body_level_STF, quat_body_level_STF, vel_base_link_STF, acc_body_level_STF,
                                                pos_desired_stance_foot_STF, quat_desired_stance_foot_STF, vel_desired_stance_foot_STF, acc_desired_stance_foot_STF,
                                                pos_desired_swing_foot_STF, quat_desired_swing_foot_STF, vel_desired_swing_foot_STF, acc_desired_swing_foot_STF);
        }

        // Marker publishers
        int marker_type;
        marker_type = visualization_msgs::msg::Marker::SPHERE;
        publish_marker(marker_type, next_footstep_STF, "next_footstep", "STF", 1, Eigen::Vector3d(1.0, 0.0, 1.0), pub_marker_next_footstep_);
        // pub safety circle around the STF
        std::list<Eigen::Vector3d> safety_circle_points;
        for (int i = 0; i <= 2 * M_PI / 0.1; i++)
        {
            Eigen::Vector3d safety_circle_point;
            safety_circle_point << robot_params.safety_radius_CP * cos(i * 0.1), robot_params.safety_radius_CP * sin(i * 0.1), 0.0;
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
            if (dt > 0.1)
            {
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

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_body_trajectory_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_next_footstep_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_next_safe_footstep_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_dcm_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_desired_dcm_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_stance_foot_BF_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_swing_foot_BF_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_markers_foot_traj_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_markers_safety_circle_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_vel_BF_;

    rclcpp::Publisher<biped_bringup::msg::StampedBool>::SharedPtr pub_desired_left_contact_;
    rclcpp::Publisher<biped_bringup::msg::StampedBool>::SharedPtr pub_desired_right_contact_;

    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_state_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr contact_right_sub_;
    rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr contact_left_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_cmd_sub_;

    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped> pub_swing_foot_BF_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped> pub_stance_foot_BF_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped> pub_dcm_desired_BF_;

    rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr clock_sub_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string r_foot_frame_id_;
    std::string l_foot_frame_id_;
    std::string r_foot_urdf_frame_id_;
    std::string l_foot_urdf_frame_id_;
    std::string base_link_frame_id_;

    Eigen::Transform<double, 3, Eigen::AffineCompact> T_STF_to_BLF_;
    Eigen::Vector3d previous_vel_base_link_STF_;


    struct
    {
        double robot_height;
        double t_step;
        double omega;
        double dt_ctrl;
        double duration_init_traj;
        double safety_radius_CP;
        double T_contact_ignore;
    } robot_params;

    bool foot_right_contact_;
    bool foot_left_contact_;

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
    float time_since_last_step_;
    float remaining_time_in_step_;

    Eigen::Vector3d vel_d_;

    std::string state_;

    double timeout_for_no_feet_in_contact_;

    // init variables
    double t_init_traj_;
    std::vector<Eigen::Vector<double, 4>> coeffs_baselink_init_traj_;
    std::vector<Eigen::Vector<double, 4>> coeffs_swing_foot_init_traj_;
    bool initialization_done_;

    bool tf_for_feet_aquired_;
    Eigen::Vector3d previous_desired_stance_foot_BLF_;

    int counter_calib_traj_;

    Eigen::Vector3d previous_base_link_vel_BF_;
    FootTrajectory foot_trajectory_;
    FootTrajectory foot_stance_trajectory_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CapturePoint>());
    rclcpp::shutdown();
    return 0;
}