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

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&IKNode::joint_state_cb, this, _1));

        foot_desired_sub_ = this->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
            "foot_positions", 10, std::bind(&IKNode::foot_desired_cb, this, _1));

        // odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        //     "/odom", 10, std::bind(&IKNode::odom_cb, this, _1));

        robot_joints_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("joint_trajectory", 10);
    }

private:

    void foot_desired_cb(trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg)
    {
        if (!robot_.has_model()) {
            RCLCPP_ERROR_SKIPFIRST_THROTTLE(this->get_logger(), *this->get_clock(), 1000 /* [ms] */, "No robot model loaded");
            return;
        }
        trajectory_msgs::msg::JointTrajectory out_msg;
        out_msg.header = msg->header;
        for (const auto &pt : msg->points) {
            if (pt.transforms.size() != msg->joint_names.size()) {
                RCLCPP_ERROR(this->get_logger(), "Invalid joint trajectory point: number of transforms (%zu) does not match number of joints (%zu)", pt.transforms.size(), msg->joint_names.size());
                return;
            }
            if (pt.velocities.size() != 0 && pt.velocities.size() != msg->joint_names.size()) {
                RCLCPP_ERROR(this->get_logger(), "Invalid joint trajectory point: number of velocities (%zu) does not match number of joints (%zu)", pt.velocities.size(), msg->joint_names.size());
                return;
            }
            if (pt.accelerations.size() != 0 && pt.accelerations.size() != msg->joint_names.size()) {
                RCLCPP_ERROR(this->get_logger(), "Invalid joint trajectory point: number of accelerations (%zu) does not match number of joints (%zu)", pt.accelerations.size(), msg->joint_names.size());
                return;
            }

            std::vector<IKRobot::BodyState> bodies;
            // if base_link is missing, create it at zero
            if (std::find(msg->joint_names.begin(), msg->joint_names.end(), "base_link") == msg->joint_names.end()) {
                if (msg->header.frame_id != "base_link") {
                    RCLCPP_ERROR(this->get_logger(), "base_link missing in MultiDofJoints and frame_id is not base_link");
                    return;
                }
                bodies.push_back(IKRobot::BodyState("base_link", Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()));
            }
            for (size_t i = 0; i < msg->joint_names.size(); i++) {
                const auto &name = msg->joint_names[i];
                const auto &pos = Eigen::Vector3d(pt.transforms[i].translation.x, pt.transforms[i].translation.y, pt.transforms[i].translation.z);
                const auto &rot = Eigen::Quaterniond(pt.transforms[i].rotation.w, pt.transforms[i].rotation.x, pt.transforms[i].rotation.y, pt.transforms[i].rotation.z);
                IKRobot::BodyState body_state(name, pos, rot);
                // TODO add velocities and accelerations if they are present
                if (!this->has_parameter(name + ".joint_type")) {
                    this->declare_parameter(name + ".joint_type", "FULL_6DOF");
                }
                std::string joint_type = this->get_parameter(name + ".joint_type").as_string();
                if (joint_type == "POS_AXIS" && !this->has_parameter(name + ".axis")) {
                    this->declare_parameter(name + ".axis", std::vector<double>{1, 0, 0});
                }
                if (joint_type == "POS_AXIS") {
                    body_state.type = IKRobot::BodyState::ContraintType::POS_AXIS;
                    std::vector<double> axis = this->get_parameter(name + ".axis").as_double_array();
                    body_state.align_axis = Eigen::Vector3d(axis[0], axis[1], axis[2]);
                } else if (joint_type == "FULL_6DOF") {
                    body_state.type = IKRobot::BodyState::ContraintType::FULL_6DOF;
                } else if (joint_type == "POS_ONLY") {
                    body_state.type = IKRobot::BodyState::ContraintType::POS_ONLY;
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Invalid joint constraint type parameter: %s", joint_type.c_str());
                    return;
                }
                bodies.push_back(body_state);
            }
            std::cout << "solving with bodies : " << std::endl;
            for (const auto &body : bodies) {
                std::cout << "   " << body.name << std::endl;
            }
            std::vector<IKRobot::JointState> joint_states = this->robot_.solve(bodies);
            out_msg.joint_names.resize(joint_states.size());
            for (size_t i = 0; i < joint_states.size(); i++) {
                out_msg.joint_names[i] = joint_states[i].name;
            }
            trajectory_msgs::msg::JointTrajectoryPoint out_pt;
            out_pt.positions.resize(joint_states.size());
            for (size_t i = 0; i < joint_states.size(); i++) {
                out_pt.positions[i] = joint_states[i].position;
            }
            // todo check if velocities are present and copy them
            out_pt.time_from_start = pt.time_from_start;
            out_msg.points.push_back(out_pt);
        }
        this->robot_joints_pub_->publish(out_msg);
    }

    void joint_state_cb(sensor_msgs::msg::JointState::SharedPtr msg)
    {
        (void)msg;
        // std::cout << q_current.size() << std::endl;
        // if (q_current.size() != 0)
        // {
        //     int size_q = robot_.get_size_q();
        //     if (size_q != msg->position.size() + offset_pos_quat)
        //     {
        //         throw std::invalid_argument("Size of joint state does not match robot description");
        //     }

        //     for (int i = 0; i < size_q - offset_pos_quat; i++)
        //     {
        //         joint_names[i] = msg->name[i];
        //         q_current(7 + i) = msg->position[i];
        //     }
        // }
        // else
        // {
        //     throw std::invalid_argument("Is Robot Description Running?");
        // }
    }

    void robot_desc_cb(const std_msgs::msg::String::SharedPtr msg)
    {
        std::cout << "Subscribed to robot description" << std::endl;
        robot_.build_model(msg->data.c_str());
    }

private:
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr robot_joints_pub_;
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