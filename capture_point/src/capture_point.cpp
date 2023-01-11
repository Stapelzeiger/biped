#include <memory>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "linux_gpio/msg/stamped_bool.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "eigen3/Eigen/Dense"

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
        this->declare_parameter<double>("robot_height");
        this->declare_parameter<double>("t_step");
        this->declare_parameter<double>("dt_ctrl");
        robot_params.robot_height = this->get_parameter("robot_height").as_double();
        robot_params.t_step = this->get_parameter("t_step").as_double();
        robot_params.dt_ctrl = this->get_parameter("dt_ctrl").as_double();
        use_sim_time = this->get_parameter("use_sim_time").as_bool();
        robot_params.omega = sqrt(9.81/robot_params.robot_height);

        feet.names.push_back("FR_FOOT");
        feet.names.push_back("FL_FOOT");
        feet.contact.push_back(false);
        feet.contact.push_back(false);

        initialization_flag = false;
        stance_foot_name = "FR_FOOT";
        swing_foot_name = "FL_FOOT";
        time_since_last_step = robot_params.t_step/2;
        remaining_time_in_step = robot_params.t_step - time_since_last_step;

        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", 10, std::bind(&CapturePoint::odometry_callback, this, _1));
        
        contact_sub_ = this->create_subscription<linux_gpio::msg::StampedBool>(
            "/contact", 10, std::bind(&CapturePoint::contact_callback, this, _1));
        pub_feet_trajectory_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/foot_positions", 10);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        
        timer_ = this->create_wall_timer(10ms, std::bind(&CapturePoint::timer_callback, this)); // todo add the dt_ctrl
    }

private:
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

    void contact_callback(linux_gpio::msg::StampedBool::SharedPtr msg)
    {   
        if (feet.names.size() != msg->names.size()) {
            RCLCPP_ERROR(this->get_logger(), "Number of read contact (%zu) and  number of actual contacts (%zu) don't match", feet.names.size(), msg->names.size());
        }
        for (long unsigned int i = 0; i < feet.names.size(); i++)
        {
            auto iter_in_msg = find(msg->names.begin(), msg->names.end(), feet.names[i]);
            feet.contact[i] = msg->data[iter_in_msg - msg->names.begin()];
        }
    }

    geometry_msgs::msg::TransformStamped get_transform(std::string fromFrameRel, std::string toFrameRel)
    {
        geometry_msgs::msg::TransformStamped t;
        try {
          t = tf_buffer_->lookupTransform(
            toFrameRel, fromFrameRel,
            tf2::TimePointZero, tf2::durationFromSec(1.0));
        } catch (const tf2::TransformException & ex) {
          RCLCPP_INFO(
            this->get_logger(), "Could not transform %s to %s: %s",
            fromFrameRel.c_str(), toFrameRel.c_str(), ex.what());
        }
        return t;
    }

    Eigen::Vector3d get_dcm_STF(std::string stance_foot_name)
    {
        std::string fromFrameRel = "base_link";
        std::string toFrameRel = stance_foot_name;
        geometry_msgs::msg::TransformStamped t = get_transform(fromFrameRel, toFrameRel);

        Eigen::Vector3d dcm_STF;
        dcm_STF(0) = t.transform.translation.x + 1.0/robot_params.omega*base_link_odom.linear_velocity(0);
        dcm_STF(1) = t.transform.translation.y + 1.0/robot_params.omega*base_link_odom.linear_velocity(1);
        dcm_STF(2) = t.transform.translation.z + 1.0/robot_params.omega*base_link_odom.linear_velocity(2);

        return dcm_STF;
    }
    
    void set_initial_configuration()
    {

        Eigen::Vector3d pos_right_foot = Eigen::Vector3d(0.0, 0.0, -robot_params.robot_height);
        Eigen::Quaterniond quat_right_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        Eigen::Vector3d pos_left_foot = Eigen::Vector3d(0.0, 0.2, -robot_params.robot_height + 0.1);
        Eigen::Quaterniond quat_left_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        publish_foot_trajectories(pos_right_foot, quat_right_foot, pos_left_foot, quat_left_foot);
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

    void timer_callback()
    {


        // when run on the actual robot
        if (!use_sim_time){
            std::cout << "Testing on actual robot!" << std::endl; 
            if (std::all_of(feet.contact.begin(), feet.contact.end(), [](bool c) { return !c; }) && initialization_flag == false)
            {
                std::cout << "initialization started" << std::endl;
                set_initial_configuration();

                if (std::any_of(feet.contact.begin(), feet.contact.end(), [](bool c) { return c; }))
                {
                    initialization_flag = true;
                    std::cout << "initialization complete" << std::endl;
                }
            }
        }

        float T_contact_ignore = 0.1;

        // swing foot impact
        auto iter = find(feet.names.begin(), feet.names.end(), swing_foot_name);
        auto contact_swing_foot = feet.contact[iter - feet.names.begin()];
        
        // if (time_since_last_step > T_contact_ignore && contact_swing_foot == true){
        //     stance_foot_name = swing_foot_name;
        //     swing_foot_name = stance_foot_name;

        //     time_since_last_step = 0;

        // }

        Eigen::Vector3d dcm_STF = get_dcm_STF(stance_foot_name);

        Eigen::Vector3d next_footstep_STF;
        Eigen::Vector3d dcm_desired_STF;
        dcm_desired_STF << 0.0, 0.0, 0.0;

        next_footstep_STF = -dcm_desired_STF + dcm_STF*exp(robot_params.omega*remaining_time_in_step);

        geometry_msgs::msg::TransformStamped next_footstep_STF_msg;
        next_footstep_STF_msg.header.stamp = this->get_clock()->now();
        next_footstep_STF_msg.header.frame_id = stance_foot_name;
        next_footstep_STF_msg.child_frame_id = "next_footstep_STF";
        next_footstep_STF_msg.transform.translation.x = next_footstep_STF(0);
        next_footstep_STF_msg.transform.translation.y = next_footstep_STF(1);
        next_footstep_STF_msg.transform.translation.z = next_footstep_STF(2);
        tf_broadcaster_->sendTransform(next_footstep_STF_msg);

        std::string fromFrameRel = "next_footstep_STF";
        std::string toFrameRel = "base_link";
        geometry_msgs::msg::TransformStamped t = get_transform(fromFrameRel, toFrameRel);

        remaining_time_in_step = robot_params.t_step - time_since_last_step;
        time_since_last_step = time_since_last_step + robot_params.dt_ctrl;

        Eigen::Vector3d pos_right_foot = Eigen::Vector3d(0.0, 0.0, -robot_params.robot_height);
        Eigen::Quaterniond quat_right_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        Eigen::Vector3d pos_left_foot = Eigen::Vector3d(0.0, 0.2, -robot_params.robot_height + 0.1);
        Eigen::Quaterniond quat_left_foot = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        publish_foot_trajectories(pos_right_foot, quat_right_foot, pos_left_foot, quat_left_foot);

    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_feet_trajectory_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<linux_gpio::msg::StampedBool>::SharedPtr contact_sub_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    struct {
        float robot_height;
        float t_step;
        float omega;
        float dt_ctrl;
    } robot_params;

    struct {
        std::vector<bool> contact;
        std::vector<std::string> names;
    } feet;

    struct {
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d linear_velocity;
        Eigen::Vector3d angular_velocity;
        Eigen::Vector3d linear_acceleration;
        Eigen::Vector3d angular_acceleration;
    } base_link_odom;

    std::string stance_foot_name;
    std::string swing_foot_name;
    float time_since_last_step;
    float remaining_time_in_step;

    bool initialization_flag;
    bool use_sim_time;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CapturePoint>());
    rclcpp::shutdown();
    return 0;
}