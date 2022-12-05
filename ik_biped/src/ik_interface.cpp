#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "ik_class_pin.hpp"
#include <math.h>

using namespace std::placeholders;

const int offset_pos_quat = 7;

class IKNode : public rclcpp::Node
{

public:
    IKNode() : Node("ik_node")
    {
        robot_desc_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/robot_description", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(), std::bind(&IKNode::robot_desc_cb, this, _1));

        RCLCPP_INFO(this->get_logger(), "Waiting to get the robot description");
        rclcpp::WaitSet wait_set;
        wait_set.add_subscription(robot_desc_sub_);
        auto ret = wait_set.wait(std::chrono::seconds(100));
        while (ret.kind() != rclcpp::WaitResultKind::Ready)
        {
            std::cout << "Still waiting to get robot description ... " << std::endl;
        }

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&IKNode::joint_state_cb, this, _1));

        foot_desired_sub_ = this->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
            "/foot_position_BL", 10, std::bind(&IKNode::foot_desired_cb, this, _1));

        // odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        //     "/odom", 10, std::bind(&IKNode::odom_cb, this, _1));

        robot_joints_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("/joint_trajectory", 10);
    }

private:
    void pub_robot_joints()
    {
        Eigen::VectorXd q_des;

        if (q_current.size() != 0)
        {
            q_current(0) = 0;
            q_current(1) = 0;
            q_current(2) = 0;
            q_current(3) = 0;
            q_current(4) = 0;
            q_current(5) = 0;
            q_current(6) = 1;

            q_des = robot_.get_desired_q(q_current, translation_foot_des_BL, yaw_angle_foot_BL_des, name_pos_des_BL);
            trajectory_msgs::msg::JointTrajectory message;
            message.points.resize(joint_names.size());
            for (unsigned int i = 0; i < joint_names.size(); i++)
            {
                message.joint_names.push_back(joint_names[i]);
                message.points[i].positions.push_back(q_des[offset_pos_quat + i]);
                message.points[i].velocities.push_back(0.0);
                message.points[i].accelerations.push_back(0.0);
                message.points[i].effort.push_back(0.0);
            }
            robot_joints_pub_->publish(message);
        }
    }

    // void odom_cb(nav_msgs::msg::Odometry::SharedPtr msg)
    // {

    //     if (q_current.size() != 0)
    //     {
    //         // q_current(0) = msg->pose.pose.position.x;
    //         // q_current(1) = msg->pose.pose.position.y;
    //         // q_current(2) = msg->pose.pose.position.z;
    //         // q_current(3) = msg->pose.pose.orientation.x;
    //         // q_current(4) = msg->pose.pose.orientation.y;
    //         // q_current(5) = msg->pose.pose.orientation.z;
    //         // q_current(6) = msg->pose.pose.orientation.w;
    //     }
    //     // else
    //     // {
    //     //     throw std::invalid_argument("Is Robot Description Running?");
    //     // }
    // }

    void foot_desired_cb(trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg)
    {
        std::cout << "foot desired CB!" << std::endl;
        name_pos_des_BL = msg->joint_names[0];
        Eigen::Quaterniond rotation_foot_des_BL;

        translation_foot_des_BL(0) = msg->points[0].transforms[0].translation.x;
        translation_foot_des_BL(1) = msg->points[0].transforms[0].translation.y;
        translation_foot_des_BL(2) = msg->points[0].transforms[0].translation.z;
        rotation_foot_des_BL.x() = msg->points[0].transforms[0].rotation.x;
        rotation_foot_des_BL.y() = msg->points[0].transforms[0].rotation.y;
        rotation_foot_des_BL.z() = msg->points[0].transforms[0].rotation.z;
        rotation_foot_des_BL.w() = msg->points[0].transforms[0].rotation.w;

        Eigen::Vector3d axis_in_BLF = rotation_foot_des_BL*Eigen::Vector3d(1, 0, 0);

        yaw_angle_foot_BL_des = atan2(axis_in_BLF(1), axis_in_BLF(0));

        if (q_current.size() != 0)
        {
            // std::cout << translation_foot_des_BL << std::endl;
            // std::cout << "Yaw angle:" << yaw_angle_foot_BL_des << std::endl;
            pub_robot_joints();
        }
    }

    void joint_state_cb(sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (q_current.size() != 0)
        {
            int size_q = robot_.get_size_q();
            for (int i = 0; i < size_q - offset_pos_quat; i++)
            {
                joint_names[i] = msg->name[i];
                q_current(7 + i) = msg->position[i];
            }
        }
        // else
        // {
        //     throw std::invalid_argument("Is Robot Description Running?");
        // }
    }

    void robot_desc_cb(const std_msgs::msg::String::SharedPtr msg)
    {
        std::cout << "Subscribed to robot description" << std::endl;
        robot_.build_model(msg->data.c_str());
        int size_q = robot_.get_size_q();
        q_current.resize(size_q);
        joint_names.resize(size_q - offset_pos_quat);
        std::cout << "size_q" << robot_.get_size_q() << std::endl;
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr robot_joints_pub_;
    size_t count_;

    Eigen::VectorXd q_current;
    std::vector<std::string> joint_names;
    Eigen::Vector3d translation_foot_des_BL;
    double yaw_angle_foot_BL_des;

    std::string name_pos_des_BL;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_desc_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr foot_desired_sub_;
    // rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    IKRobot robot_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IKNode>());
    rclcpp::shutdown();
    return 0;
}