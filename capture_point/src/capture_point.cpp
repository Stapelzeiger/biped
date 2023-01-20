#include <memory>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"

#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "biped_bringup/msg/stamped_bool.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "rosgraph_msgs/msg/clock.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "eigen3/Eigen/Dense"

#include "foot_trajectory.h"

using namespace std::placeholders;
using namespace std::chrono_literals;

class CapturePoint : public rclcpp::Node
{

public:
    CapturePoint() : Node("cp_node")
    {
        initialization_flag_ = false;
        set_initial_parameters();

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        pub_feet_trajectory_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/foot_positions", 10);

        pub_markers_array_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/markers_array", 10);

        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", 10, std::bind(&CapturePoint::odometry_callback, this, _1));

        contact_right_sub_ = this->create_subscription<biped_bringup::msg::StampedBool>(
            "/poll_FR_FOOT/gpio", 10, std::bind(&CapturePoint::contact_right_callback, this, _1));

        contact_left_sub_ = this->create_subscription<biped_bringup::msg::StampedBool>(
            "/poll_FL_FOOT/gpio", 10, std::bind(&CapturePoint::contact_left_callback, this, _1));

        std::chrono::duration<double> period = robot_params.dt_ctrl * 1s;
        timer_ = rclcpp::create_timer(this, this->get_clock(), period, std::bind(&CapturePoint::timer_callback, this));
    }

private:
    void set_initial_parameters()
    {
        this->declare_parameter<double>("robot_height");
        this->declare_parameter<double>("t_step");
        this->declare_parameter<double>("ctrl_time_sec");

        robot_params.robot_height = this->get_parameter("robot_height").as_double();
        robot_params.t_step = this->get_parameter("t_step").as_double();
        robot_params.dt_ctrl = this->get_parameter("ctrl_time_sec").as_double();
        robot_params.omega = sqrt(9.81 / robot_params.robot_height);
    }

    void set_initial_configuration()
    {
        Eigen::Vector3d desired_stance_foot_pos_BF = Eigen::Vector3d(0.0, 0.0, -robot_params.robot_height);
        Eigen::Quaterniond stance_foot_quat = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        Eigen::Vector3d desired_swing_foot_pos_BF = Eigen::Vector3d(0.0, 0.2, -robot_params.robot_height + 0.1);
        Eigen::Quaterniond swing_foot_quat = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        publish_foot_trajectories(desired_stance_foot_pos_BF, stance_foot_quat, desired_swing_foot_pos_BF, swing_foot_quat);

        swing_foot_is_left_ = true;
        foot_right_contact_ = false;
        foot_left_contact_ = false;

        time_since_last_step_ = robot_params.t_step / 2;
        remaining_time_in_step_ = robot_params.t_step - time_since_last_step_;
        timeout_for_initialization_ = 0;
    
        std::string swing_foot_name;
        std::string stance_foot_name;
        swing_foot_name = "FL_FOOT";
        stance_foot_name = "FR_FOOT";

        computed_swing_foot_pos_STF_ = desired_swing_foot_pos_BF - desired_stance_foot_pos_BF;
        computed_swing_foot_vel_STF_ << 0.0, 0.0, 0.0;
        swing_foot_position_beginning_of_step_STF_ << 0.0, 0.0, 0.0;

        Eigen::Vector3d stance_foot_BF = get_eigen_transform(stance_foot_name, "base_link").translation();
        Eigen::Vector3d swing_foot_BF = get_eigen_transform(swing_foot_name, "base_link").translation();        
        Eigen::Quaterniond quad_unit = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        Eigen::Vector3d color_sw_foot; color_sw_foot << 0.0, 1.0, 0.0;
        Eigen::Vector3d color_st_foot; color_st_foot << 1.0, 0.0, 0.0;
        publish_markers(swing_foot_BF, quad_unit, "swing_foot_BF", "base_link", 0, color_sw_foot);
        publish_markers(stance_foot_BF, quad_unit, "stance_foot_BF", "base_link", 1, color_st_foot);

    }

    void odometry_callback(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        base_link_odom.position(0) = msg->pose.pose.position.x;
        base_link_odom.position(1) = msg->pose.pose.position.y;
        base_link_odom.position(2) = msg->pose.pose.position.z;
        base_link_odom.orientation.w() = msg->pose.pose.orientation.w;
        base_link_odom.orientation.x() = msg->pose.pose.orientation.x;
        base_link_odom.orientation.y() = msg->pose.pose.orientation.y;
        base_link_odom.orientation.z() = msg->pose.pose.orientation.z;
        base_link_odom.linear_velocity(0) = msg->twist.twist.linear.x;
        base_link_odom.linear_velocity(1) = msg->twist.twist.linear.y;
        base_link_odom.linear_velocity(2) = msg->twist.twist.linear.z;
        base_link_odom.angular_velocity(0) = msg->twist.twist.angular.x;
        base_link_odom.angular_velocity(1) = msg->twist.twist.angular.y;
        base_link_odom.angular_velocity(2) = msg->twist.twist.angular.z;
    }

    void timer_callback()
    {
        std::cout << foot_left_contact_ << " " << foot_right_contact_ << std::endl;
        
        if (foot_left_contact_ == false and foot_right_contact_ == false)
        {
            timeout_for_initialization_ -= robot_params.dt_ctrl;
        } else {
            timeout_for_initialization_ = 0.5;
        }
        if (timeout_for_initialization_ < 0)
        {
            set_initial_configuration();
        } else {
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
            swing_foot_name = "FL_FOOT";
            stance_foot_name = "FR_FOOT";
        } else {
            swing_foot_contact = foot_right_contact_;
            swing_foot_name = "FR_FOOT";
            stance_foot_name = "FL_FOOT";
        }

        Eigen::Transform<double, 3, Eigen::AffineCompact> T_STF_to_BLF;
        auto T_IF_to_BF = get_eigen_transform("odom", "base_link");
        auto T_BLF_to_BF = get_BLF_to_BF();
        broadcast_transform("base_link", "BLF", T_BLF_to_BF.translation(), Eigen::Quaterniond(T_BLF_to_BF.rotation()));
        auto T_BF_to_BLF = T_BLF_to_BF.inverse();
        
        float T_contact_ignore = 0.1;
        if (time_since_last_step_ > T_contact_ignore && swing_foot_contact == true)
        {
            swing_foot_is_left_ = !swing_foot_is_left_;
            std::swap(stance_foot_name, swing_foot_name);

            Eigen::Vector3d stance_foot_BF = get_eigen_transform(stance_foot_name, "base_link").translation();
            Eigen::Vector3d swing_foot_BF = get_eigen_transform(swing_foot_name, "base_link").translation();
            
            auto swing_foot_BLF = T_BF_to_BLF * swing_foot_BF;
            auto swing_foot_STF = T_STF_to_BLF.inverse() * swing_foot_BLF;
            computed_swing_foot_pos_STF_ = swing_foot_STF;
            computed_swing_foot_vel_STF_ << 0.0, 0.0, 0.0;
            swing_foot_position_beginning_of_step_STF_ = computed_swing_foot_pos_STF_;

            std::cout << "-------------------------- switch! (miau)";
            std::cout << "time_since_last_step = " << time_since_last_step_ << std::endl;
            time_since_last_step_ = 0;
        }


        Eigen::Vector3d stance_foot_BF = get_eigen_transform(stance_foot_name, "base_link").translation();
        Eigen::Vector3d stance_foot_BLF = T_BF_to_BLF * stance_foot_BF;
        T_STF_to_BLF.linear() = Eigen::Matrix3d::Identity();
        T_STF_to_BLF.translation() = stance_foot_BLF;
        broadcast_transform("BLF", "STF", T_STF_to_BLF.translation(), Eigen::Quaterniond(T_STF_to_BLF.rotation()));

        Eigen::Vector3d dcm_desired_STF;
        dcm_desired_STF << 0.1, 0.0, 0.0;

        Eigen::Vector3d base_link_vel_BF;
        base_link_vel_BF << base_link_odom.linear_velocity(0), base_link_odom.linear_velocity(1), base_link_odom.linear_velocity(2);
        Eigen::Vector3d base_link_vel_BLF = T_BF_to_BLF.rotation() * base_link_vel_BF;
        Eigen::Vector3d base_link_vel_STF = T_STF_to_BLF.inverse().rotation() * base_link_vel_BLF;
        Eigen::Vector3d dcm_STF;
        dcm_STF(0) = T_STF_to_BLF.inverse().translation()[0] + 1.0 / robot_params.omega * base_link_vel_STF(0);
        dcm_STF(1) = T_STF_to_BLF.inverse().translation()[1] + 1.0 / robot_params.omega * base_link_vel_STF(1);
        dcm_STF(2) = 0;

        Eigen::Quaterniond quat_identity = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        publish_markers(dcm_STF, quat_identity, "dcm_STF", "STF", 2, Eigen::Vector3d(0.0, 0.0, 1.0));

        Eigen::Vector3d next_footstep_STF;
        next_footstep_STF = -dcm_desired_STF + dcm_STF * exp(robot_params.omega * remaining_time_in_step_);
        Eigen::Vector3d des_pos_foot_STF;
        des_pos_foot_STF << next_footstep_STF(0), next_footstep_STF(1), 0;
        publish_markers(next_footstep_STF, quat_identity, "next_footstep_STF", "STF", 2, Eigen::Vector3d(1.0, 0.0, 1.0));

        desired_swing_foot_pos_vel_acc_STF = get_swing_foot_pos_vel(
            time_since_last_step_,
            robot_params.t_step,
            computed_swing_foot_pos_STF_,
            computed_swing_foot_vel_STF_,
            swing_foot_position_beginning_of_step_STF_,
            des_pos_foot_STF);


        for (int i = 0; i < 3; i++)
        {
            computed_swing_foot_pos_STF_(i) = desired_swing_foot_pos_vel_acc_STF.vel(i) * robot_params.dt_ctrl + desired_swing_foot_pos_vel_acc_STF.pos(i);
            computed_swing_foot_vel_STF_(i) = desired_swing_foot_pos_vel_acc_STF.acc(i) * robot_params.dt_ctrl + desired_swing_foot_pos_vel_acc_STF.vel(i);
        }

        remaining_time_in_step_ = robot_params.t_step - time_since_last_step_;

        time_since_last_step_ = time_since_last_step_ + robot_params.dt_ctrl;

        Eigen::Transform transform_stance_foot_BF = get_eigen_transform(stance_foot_name, "base_link");

        Eigen::Vector3d desired_swing_foot_pos_BLF;
        desired_swing_foot_pos_BLF = T_STF_to_BLF * desired_swing_foot_pos_vel_acc_STF.pos;

        if (swing_foot_name == "FR_FOOT")
        {
            Eigen::Vector3d pos_right_foot = Eigen::Vector3d(desired_swing_foot_pos_BLF(0),
                                                             desired_swing_foot_pos_BLF(1),
                                                             desired_swing_foot_pos_BLF(2));
            Eigen::Quaterniond quat_right_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            Eigen::Vector3d pos_left_foot = Eigen::Vector3d(transform_stance_foot_BF.translation()[0], transform_stance_foot_BF.translation()[1], -robot_params.robot_height);
            Eigen::Quaterniond quat_left_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            publish_foot_trajectories(pos_right_foot, quat_right_foot, pos_left_foot, quat_left_foot);
        }
        else
        {
            Eigen::Vector3d pos_left_foot = Eigen::Vector3d(desired_swing_foot_pos_BLF(0),
                                                            desired_swing_foot_pos_BLF(1),
                                                            desired_swing_foot_pos_BLF(2));
            Eigen::Quaterniond quat_left_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            Eigen::Vector3d pos_right_foot = Eigen::Vector3d(transform_stance_foot_BF.translation()[0], transform_stance_foot_BF.translation()[1], -robot_params.robot_height);
            Eigen::Quaterniond quat_right_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            publish_foot_trajectories(pos_right_foot, quat_right_foot, pos_left_foot, quat_left_foot);
        }
    }

    void contact_right_callback(biped_bringup::msg::StampedBool::SharedPtr msg)
    {
        foot_right_contact_ = msg->data;
    }

    void contact_left_callback(biped_bringup::msg::StampedBool::SharedPtr msg)
    {
        foot_left_contact_ = msg->data;
    }


    Eigen::Transform<double, 3, Eigen::AffineCompact> get_BLF_to_BF()
    {
        Eigen::Transform<double, 3, Eigen::AffineCompact> transform_BF_to_IF = get_eigen_transform("base_link", "odom");
        
        Eigen::Vector3d x_vec_BF; x_vec_BF << 1.0, 0.0, 0.0;
        Eigen::Vector3d x_vec_IF = transform_BF_to_IF.rotation() * x_vec_BF;
        Eigen::Vector3d bl_x_vec_IF; bl_x_vec_IF << x_vec_IF(0), x_vec_IF(1), 0.0;
        bl_x_vec_IF.normalize();
        Eigen::Vector3d bl_z_vec_IF; bl_z_vec_IF << 0.0, 0.0, 1.0;
        Eigen::Vector3d bl_y_vec_IF = bl_z_vec_IF.cross(bl_x_vec_IF);

        Eigen::Transform<double, 3, Eigen::AffineCompact> T_BLF_to_IF;
        T_BLF_to_IF.linear() << bl_x_vec_IF, bl_y_vec_IF, bl_z_vec_IF;
        T_BLF_to_IF.translation() = transform_BF_to_IF.translation();

        Eigen::Transform<double, 3, Eigen::AffineCompact> T_BLF_to_BF = transform_BF_to_IF.inverse() * T_BLF_to_IF;
        return T_BLF_to_BF;
    }


    geometry_msgs::msg::TransformStamped get_transform(std::string fromFrameRel, std::string toFrameRel)
    {
        geometry_msgs::msg::TransformStamped t;
        try
        {
            t = tf_buffer_->lookupTransform(
                toFrameRel, fromFrameRel,
                tf2::TimePointZero, tf2::durationFromSec(10.0));
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_INFO(
                this->get_logger(), "Could not transform %s to %s: %s",
                fromFrameRel.c_str(), toFrameRel.c_str(), ex.what());
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



    void publish_markers(Eigen::Vector3d pos, Eigen::Quaterniond quat, std::string name_marker, std::string frame_id, int id, Eigen::Vector3d color)
    {
        visualization_msgs::msg::Marker marker_msg;
        marker_msg.header.frame_id = frame_id;
        marker_msg.header.stamp = this->get_clock()->now();
        marker_msg.ns = name_marker;
        marker_msg.id = id;
        marker_msg.type = visualization_msgs::msg::Marker::SPHERE;
        marker_msg.action = visualization_msgs::msg::Marker::ADD;
        marker_msg.pose.position.x = pos[0];
        marker_msg.pose.position.y = pos[1];
        marker_msg.pose.position.z = pos[2];
        marker_msg.pose.orientation.x = quat.x();
        marker_msg.pose.orientation.y = quat.y();
        marker_msg.pose.orientation.z = quat.z();
        marker_msg.pose.orientation.w = quat.w();
        marker_msg.scale.x = 0.05;
        marker_msg.scale.y = 0.05;
        marker_msg.scale.z = 0.05;
        marker_msg.color.a = 1.0;
        marker_msg.color.r = color(0);
        marker_msg.color.g = color(1);
        marker_msg.color.b = color(2);
        marker_array.markers.push_back(marker_msg);
        pub_markers_array_->publish( marker_array );

    }

    void publish_foot_trajectories(Eigen::Vector3d pos_right_foot, Eigen::Quaterniond quat_right_foot, Eigen::Vector3d pos_left_foot, Eigen::Quaterniond quat_left_foot)
    {
        trajectory_msgs::msg::MultiDOFJointTrajectory msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";
        msg.joint_names.push_back("FR_ANKLE");
        msg.joint_names.push_back("FL_ANKLE");

        geometry_msgs::msg::Transform right_foot;
        right_foot.translation.x = pos_right_foot(0);
        right_foot.translation.y = pos_right_foot(1);
        right_foot.translation.z = pos_right_foot(2);
        right_foot.rotation.x = quat_right_foot.x();
        right_foot.rotation.y = quat_right_foot.y();
        right_foot.rotation.z = quat_right_foot.z();
        right_foot.rotation.w = quat_right_foot.w();

        geometry_msgs::msg::Transform left_foot;
        left_foot.translation.x = pos_left_foot(0);
        left_foot.translation.y = pos_left_foot(1);
        left_foot.translation.z = pos_left_foot(2);
        left_foot.rotation.x = quat_left_foot.x();
        left_foot.rotation.y = quat_left_foot.y();
        left_foot.rotation.z = quat_left_foot.z();
        left_foot.rotation.w = quat_left_foot.w();

        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint foot_pos_point;
        foot_pos_point.transforms.push_back(right_foot);
        foot_pos_point.transforms.push_back(left_foot);

        msg.points.push_back(foot_pos_point);
        pub_feet_trajectory_->publish(msg);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_feet_trajectory_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_array_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr contact_right_sub_;
    rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr contact_left_sub_;

    rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr clock_sub_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    visualization_msgs::msg::MarkerArray marker_array;

    struct
    {
        float robot_height;
        float t_step;
        float omega;
        float dt_ctrl;
    } robot_params;

    bool foot_right_contact_;
    bool foot_left_contact_;

    struct
    {
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d linear_velocity;
        Eigen::Vector3d angular_velocity;
        Eigen::Vector3d linear_acceleration;
        Eigen::Vector3d angular_acceleration;
    } base_link_odom;

    bool swing_foot_is_left_;
    
    Eigen::Vector3d computed_swing_foot_pos_STF_;
    Eigen::Vector3d computed_swing_foot_vel_STF_;
    Eigen::Vector3d swing_foot_position_beginning_of_step_STF_;

    float time_since_last_step_;
    float remaining_time_in_step_;

    bool initialization_flag_;

    float timeout_for_initialization_;
    foot_pos_vel_acc_struct desired_swing_foot_pos_vel_acc_STF;
    double counter_points_traj;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CapturePoint>());
    rclcpp::shutdown();
    return 0;
}