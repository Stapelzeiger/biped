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
#include "linux_gpio/msg/stamped_bool.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "rosgraph_msgs/msg/clock.hpp"
#include "eigen3/Eigen/Dense"

#include "foot_trajectory.h"
// subscribe to foot sensors
// subscribe to odometry
// compute DCM relative to the foot
// compute forward kinematics to get the foot position
// compute the CP location
// generate foot trajectories
// publish foot trajectories

using namespace std::placeholders;
using namespace std::chrono_literals;

class CapturePoint : public rclcpp::Node
{

public:
    CapturePoint() : Node("cp_node")
    {
        initialization_flag = false;
        set_initial_parameters();

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        pub_feet_trajectory_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/foot_positions", 10);
        pub_next_footstep_OdomF_ = this->create_publisher<geometry_msgs::msg::Vector3>("/next_footstep_OdomF", 10);
        pub_desired_swing_foot_pos_OdomF_ = this->create_publisher<geometry_msgs::msg::Vector3>("desired_swing_foot_pos_OdomF", 10);
        pub_initialization_flag_ = this->create_publisher<std_msgs::msg::Bool>("/initialization_flag", 10);

        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", 10, std::bind(&CapturePoint::odometry_callback, this, _1));

        contact_sub_ = this->create_subscription<linux_gpio::msg::StampedBool>(
            "/contact", 10, std::bind(&CapturePoint::contact_callback, this, _1));

        clock_sub_ = this->create_subscription<rosgraph_msgs::msg::Clock>(
            "/clock", 10, std::bind(&CapturePoint::clock_callback, this, _1));

        time_now_sec = this->get_clock()->now().seconds();
        counter = 0;
    }

private:
    void set_initial_parameters()
    {
        this->declare_parameter<double>("robot_height");
        this->declare_parameter<double>("t_step");
        this->declare_parameter<double>("ctrl_time_sec");
        this->declare_parameter<double>("sim_time_sec");
        robot_params.robot_height = this->get_parameter("robot_height").as_double();
        robot_params.t_step = this->get_parameter("t_step").as_double();
        robot_params.dt_ctrl = this->get_parameter("ctrl_time_sec").as_double();
        robot_params.dt_sim = this->get_parameter("sim_time_sec").as_double();
        use_sim_time = this->get_parameter("use_sim_time").as_bool();
        robot_params.omega = sqrt(9.81 / robot_params.robot_height);

        stance_foot_name = "FR_FOOT";
        swing_foot_name = "FL_FOOT";
        feet.names.push_back(stance_foot_name);
        feet.names.push_back(swing_foot_name);
        feet.contact.push_back(false);
        feet.contact.push_back(false);

        time_since_last_step = robot_params.t_step / 2;
        remaining_time_in_step = robot_params.t_step - time_since_last_step;
    }

    bool set_initial_configuration()
    {
        Eigen::Vector3d desired_stance_foot_pos_BF = Eigen::Vector3d(0.0, 0.0, -robot_params.robot_height);
        Eigen::Quaterniond stance_foot_quat = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        Eigen::Vector3d desired_swing_foot_pos_BF = Eigen::Vector3d(0.0, 0.2, -robot_params.robot_height + 0.1);
        Eigen::Quaterniond swing_foot_quat = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        publish_foot_trajectories(desired_stance_foot_pos_BF, stance_foot_quat, desired_swing_foot_pos_BF, swing_foot_quat);

        geometry_msgs::msg::TransformStamped transform_stance_foot_BF = get_transform_stamped(stance_foot_name, "base_link");
        geometry_msgs::msg::TransformStamped transform_swing_foot_BF = get_transform_stamped(swing_foot_name, "base_link");
        Eigen::Vector3d actual_stance_foot_pos_BF;
        Eigen::Vector3d actual_swing_foot_pos_BF;

        actual_stance_foot_pos_BF << transform_stance_foot_BF.transform.translation.x, transform_stance_foot_BF.transform.translation.y, transform_stance_foot_BF.transform.translation.z;
        actual_swing_foot_pos_BF << transform_swing_foot_BF.transform.translation.x, transform_swing_foot_BF.transform.translation.y, transform_swing_foot_BF.transform.translation.z;

        auto error_stance_foot_pos_BF = desired_stance_foot_pos_BF - actual_stance_foot_pos_BF;
        auto error_swing_foot_pos_BF = desired_swing_foot_pos_BF - actual_swing_foot_pos_BF;

        if (error_stance_foot_pos_BF.squaredNorm() < 0.0001 && error_swing_foot_pos_BF.squaredNorm() < 0.0001)
        {
            transform_swing_foot_STF = get_transform_stamped(swing_foot_name, stance_foot_name);
            computed_swing_foot_pos_STF << transform_swing_foot_STF.transform.translation.x, transform_swing_foot_STF.transform.translation.y, transform_swing_foot_STF.transform.translation.z;
            computed_swing_foot_vel_STF << 0.0, 0.0, 0.0;
            swing_foot_position_beginning_of_step_STF << computed_swing_foot_pos_STF(0), computed_swing_foot_pos_STF(1), computed_swing_foot_pos_STF(2);
            next_footstep_STF << 0.0, 0.0, 0.0;
            broadcast_transform(stance_foot_name, "next_footstep_STF", next_footstep_STF, Eigen::Quaterniond::Identity());
            return true;
        }
        else
        {
            return false;
        }
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

        Eigen::Vector3d pt_pos;
        pt_pos << base_link_odom.position(0), base_link_odom.position(1), base_link_odom.position(2);
        Eigen::Quaterniond pt_quat;
        pt_quat.w() = base_link_odom.orientation.w();
        pt_quat.x() = base_link_odom.orientation.x();
        pt_quat.y() = base_link_odom.orientation.y();
        pt_quat.z() = base_link_odom.orientation.z();

        broadcast_transform("odom", "base_link", pt_pos, pt_quat);
    }

    void clock_callback(rosgraph_msgs::msg::Clock msg)
    {
        auto delta_t_sim = msg.clock.sec + msg.clock.nanosec * 1e-9 - time_now_sec;
        time_now_sec = msg.clock.sec + msg.clock.nanosec * 1e-9;

        if (initialization_flag == false)
        {
            initialization_flag = set_initial_configuration();
        }
        std_msgs::msg::Bool initialization_flag_msg;
        initialization_flag_msg.data = initialization_flag;
        pub_initialization_flag_->publish(initialization_flag_msg);

        std::cout << "initialization flag" << initialization_flag << std::endl;

        if (initialization_flag == true)
        {
            double ratio = robot_params.dt_ctrl / robot_params.dt_sim;
            if (counter % int(ceil(ratio)) == 0)
            {
                run_capture_point_controller();
            }
        }
        counter = counter + 1;
    }

    void run_capture_point_controller()
    {
        float T_contact_ignore = 0.1;
        auto iter = find(feet.names.begin(), feet.names.end(), swing_foot_name);
        auto contact_swing_foot = feet.contact[iter - feet.names.begin()];

        if (time_since_last_step > T_contact_ignore && contact_swing_foot == true)
        {
            std::swap(stance_foot_name, swing_foot_name);
            transform_swing_foot_STF = get_transform_stamped(swing_foot_name, stance_foot_name);
            computed_swing_foot_pos_STF << transform_swing_foot_STF.transform.translation.x, transform_swing_foot_STF.transform.translation.y, transform_swing_foot_STF.transform.translation.z;
            computed_swing_foot_vel_STF << 0.0, 0.0, 0.0;
            swing_foot_position_beginning_of_step_STF << computed_swing_foot_pos_STF(0), computed_swing_foot_pos_STF(1), computed_swing_foot_pos_STF(2);
            // publish trajectories
            std::cout << "-------------------------- switch! ";
            std::cout << "time_since_last_step = " << time_since_last_step << std::endl;
            time_since_last_step = 0;
        }

        transform_swing_foot_STF = get_transform_stamped(swing_foot_name, stance_foot_name);

        Eigen::Vector3d dcm_desired_STF;
        dcm_desired_STF << 0.0, 0.0, 0.0;
        Eigen::Vector3d dcm_STF = get_dcm_STF(stance_foot_name);

        next_footstep_STF = -dcm_desired_STF + dcm_STF * exp(robot_params.omega * remaining_time_in_step);
        broadcast_transform(stance_foot_name, "next_footstep_STF", next_footstep_STF, Eigen::Quaterniond::Identity());

        Eigen::Vector3d des_pos_foot_STF;
        des_pos_foot_STF << next_footstep_STF(0), next_footstep_STF(1), 0;
        // std::cout << "next_footstep_STF" << next_footstep_STF << std::endl;

        geometry_msgs::msg::TransformStamped next_footstep_OdomF_ = get_transform_stamped("next_footstep_STF", "odom");
        geometry_msgs::msg::Vector3 next_footstep_OdomF_msg;
        next_footstep_OdomF_msg.x = next_footstep_OdomF_.transform.translation.x;
        next_footstep_OdomF_msg.y = next_footstep_OdomF_.transform.translation.y;
        next_footstep_OdomF_msg.z = next_footstep_OdomF_.transform.translation.z;
        pub_next_footstep_OdomF_->publish(next_footstep_OdomF_msg);

        desired_swing_foot_pos_vel_acc_STF = get_swing_foot_pos_vel(
            time_since_last_step,
            robot_params.t_step,
            computed_swing_foot_pos_STF,
            computed_swing_foot_vel_STF,
            swing_foot_position_beginning_of_step_STF,
            des_pos_foot_STF);

        broadcast_transform(stance_foot_name, "desired_swing_foot_pos", desired_swing_foot_pos_vel_acc_STF.pos, Eigen::Quaterniond::Identity());

        geometry_msgs::msg::TransformStamped desired_swing_foot_pos_BF = get_transform_stamped("desired_swing_foot_pos", "base_link");

        // geometry_msgs::msg::TransformStamped desired_swing_foot_pos_OdomF = get_transform_stamped("desired_swing_foot_pos", "odom");
        // geometry_msgs::msg::Vector3 desired_swing_foot_pos_OdomF_msg;
        // desired_swing_foot_pos_OdomF_msg.x = desired_swing_foot_pos_OdomF.transform.translation.x;
        // desired_swing_foot_pos_OdomF_msg.y = desired_swing_foot_pos_OdomF.transform.translation.y;
        // desired_swing_foot_pos_OdomF_msg.z = desired_swing_foot_pos_OdomF.transform.translation.z;
        // pub_desired_swing_foot_pos_OdomF_->publish(desired_swing_foot_pos_OdomF_msg);

        for (int i = 0; i < 3; i++)
        {
            computed_swing_foot_pos_STF(i) = desired_swing_foot_pos_vel_acc_STF.vel(i) * robot_params.dt_ctrl + desired_swing_foot_pos_vel_acc_STF.pos(i);
            computed_swing_foot_vel_STF(i) = desired_swing_foot_pos_vel_acc_STF.acc(i) * robot_params.dt_ctrl + desired_swing_foot_pos_vel_acc_STF.vel(i);
        }

        remaining_time_in_step = robot_params.t_step - time_since_last_step;

        time_since_last_step = time_since_last_step + robot_params.dt_ctrl;

        geometry_msgs::msg::TransformStamped transform_stance_foot_BF = get_transform_stamped(stance_foot_name, "base_link");

        if (swing_foot_name == "FR_FOOT")
        {
            Eigen::Vector3d pos_right_foot = Eigen::Vector3d(desired_swing_foot_pos_BF.transform.translation.x,
                                                             desired_swing_foot_pos_BF.transform.translation.y,
                                                             desired_swing_foot_pos_BF.transform.translation.z);
            Eigen::Quaterniond quat_right_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            Eigen::Vector3d pos_left_foot = Eigen::Vector3d(transform_stance_foot_BF.transform.translation.x, transform_stance_foot_BF.transform.translation.y, -robot_params.robot_height);
            Eigen::Quaterniond quat_left_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            publish_foot_trajectories(pos_right_foot, quat_right_foot, pos_left_foot, quat_left_foot);
        }
        else
        {
            Eigen::Vector3d pos_left_foot = Eigen::Vector3d(desired_swing_foot_pos_BF.transform.translation.x,
                                                            desired_swing_foot_pos_BF.transform.translation.y,
                                                            desired_swing_foot_pos_BF.transform.translation.z);
            Eigen::Quaterniond quat_left_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            Eigen::Vector3d pos_right_foot = Eigen::Vector3d(transform_stance_foot_BF.transform.translation.x, transform_stance_foot_BF.transform.translation.y, -robot_params.robot_height);
            Eigen::Quaterniond quat_right_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
            publish_foot_trajectories(pos_right_foot, quat_right_foot, pos_left_foot, quat_left_foot);
        }
    }

    void contact_callback(linux_gpio::msg::StampedBool::SharedPtr msg)
    {
        if (feet.names.size() != msg->names.size())
        {
            RCLCPP_ERROR(this->get_logger(), "Number of read contact (%zu) and  number of actual contacts (%zu) don't match", feet.names.size(), msg->names.size());
        }
        for (long unsigned int i = 0; i < feet.names.size(); i++)
        {
            auto iter_in_msg = find(msg->names.begin(), msg->names.end(), feet.names[i]);
            feet.contact[i] = msg->data[iter_in_msg - msg->names.begin()];
        }
    }

    geometry_msgs::msg::TransformStamped get_transform_stamped(std::string fromFrameRel, std::string toFrameRel)
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

    Eigen::Vector3d get_dcm_STF(std::string stance_foot_name)
    {
        std::string fromFrameRel = "base_link";
        std::string toFrameRel = stance_foot_name;
        geometry_msgs::msg::TransformStamped t = get_transform_stamped(fromFrameRel, toFrameRel);

        Eigen::Vector3d dcm_STF;
        dcm_STF(0) = t.transform.translation.x + 1.0 / robot_params.omega * base_link_odom.linear_velocity(0);
        dcm_STF(1) = t.transform.translation.y + 1.0 / robot_params.omega * base_link_odom.linear_velocity(1);
        dcm_STF(2) = t.transform.translation.z + 1.0 / robot_params.omega * base_link_odom.linear_velocity(2);

        return dcm_STF;
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

    // rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_feet_trajectory_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr pub_next_footstep_OdomF_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr pub_desired_swing_foot_pos_OdomF_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_initialization_flag_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<linux_gpio::msg::StampedBool>::SharedPtr contact_sub_;
    rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr clock_sub_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    struct
    {
        float robot_height;
        float t_step;
        float omega;
        float dt_ctrl;
        float dt_sim;
    } robot_params;

    struct
    {
        std::vector<bool> contact;
        std::vector<std::string> names;
    } feet;

    struct
    {
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d linear_velocity;
        Eigen::Vector3d angular_velocity;
        Eigen::Vector3d linear_acceleration;
        Eigen::Vector3d angular_acceleration;
    } base_link_odom;

    std::string stance_foot_name;
    std::string swing_foot_name;
    geometry_msgs::msg::TransformStamped transform_swing_foot_STF;
    Eigen::Vector3d computed_swing_foot_pos_STF;
    Eigen::Vector3d computed_swing_foot_vel_STF;
    Eigen::Vector3d swing_foot_position_beginning_of_step_STF;

    float time_since_last_step;
    float remaining_time_in_step;

    bool initialization_flag;
    bool use_sim_time;
    double time_now_sec;
    long int counter;

    Eigen::Vector3d next_footstep_STF;
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