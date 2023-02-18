#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "ik_class_pin.hpp"
#include "biped_bringup/msg/stamped_bool.hpp"
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

        foot_desired_sub_ = this->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
            "foot_positions", 10, std::bind(&IKNode::foot_desired_cb, this, _1));

        contact_right_sub_ = this->create_subscription<biped_bringup::msg::StampedBool>(
            "~/contact_foot_right", 10, std::bind(&IKNode::contact_right_callback, this, _1));

        contact_left_sub_ = this->create_subscription<biped_bringup::msg::StampedBool>(
            "~/contact_foot_left", 10, std::bind(&IKNode::contact_left_callback, this, _1));


        robot_joints_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("joint_trajectory", 10);
        markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/markers", 10);
    }

private:


    void contact_right_callback(biped_bringup::msg::StampedBool::SharedPtr msg)
    {
        auto now = this->get_clock()->now();
        double dt = (now - rclcpp::Time(msg->header.stamp)).seconds();
        if (dt > 0.1)
        {
            RCLCPP_WARN(this->get_logger(), "contact right footsensor data is too old: %f", dt);
        }

        foot_right_contact_ = msg->data;
    }

    void contact_left_callback(biped_bringup::msg::StampedBool::SharedPtr msg)
    {
        auto now = this->get_clock()->now();
        double dt = (now - rclcpp::Time(msg->header.stamp)).seconds();
        if (dt > 0.1)
        {
            RCLCPP_WARN(this->get_logger(), "contact left footsensor data is too old: %f", dt);
        }

        foot_left_contact_ = msg->data;
    }

    void foot_desired_cb(trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg)
    {
        if (!robot_.has_model()) {
            RCLCPP_ERROR_SKIPFIRST_THROTTLE(this->get_logger(), *this->get_clock(), 1000 /* [ms] */, "No robot model loaded");
            return;
        }
        trajectory_msgs::msg::JointTrajectory out_msg;
        out_msg.header = msg->header;
        int pt_idx = 0;
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

                // not nice code
                if (name == "L_ANKLE"){
                    body_state.in_contact = foot_left_contact_;
                }
                if (name == "R_ANKLE"){
                    body_state.in_contact = foot_right_contact_;
                }

                bodies.push_back(body_state);
            }
            RCLCPP_DEBUG_STREAM(this->get_logger(), "solving with bodies :");
            for (const auto &body : bodies) {
                RCLCPP_DEBUG_STREAM(this->get_logger(), "   " << body.name);
            }

            std::vector<Eigen::Vector3d> body_positions_solution;
            std::vector<IKRobot::JointState> joint_states = this->robot_.solve(bodies, body_positions_solution);
            out_msg.joint_names.resize(joint_states.size());
            for (size_t i = 0; i < joint_states.size(); i++) {
                out_msg.joint_names[i] = joint_states[i].name;
            }
            trajectory_msgs::msg::JointTrajectoryPoint out_pt;
            out_pt.positions.resize(joint_states.size());

            for (size_t i = 0; i < joint_states.size(); i++) {
                out_pt.positions[i] = joint_states[i].position;
            }

            out_pt.effort.resize(joint_states.size()); // todo make this more robust
            for (size_t i = 0; i < joint_states.size(); i++) {
                out_pt.effort[i] = joint_states[i].effort;
            }

            out_pt.time_from_start = pt.time_from_start;
            out_msg.points.push_back(out_pt);

            if (markers_pub_->get_subscription_count() > 0) {
                visualization_msgs::msg::MarkerArray marker_array_msg;
                marker_array_msg.markers.resize(body_positions_solution.size() * 2);
                assert(body_positions_solution.size() == bodies.size());
                assert(body_positions_solution.size() == pt.transforms.size());
                for (unsigned i = 0; i < body_positions_solution.size(); i++)
                {
                    marker_array_msg.markers[i].header = msg->header;
                    marker_array_msg.markers[i].ns = bodies[i].name;
                    marker_array_msg.markers[i].id = pt_idx;
                    marker_array_msg.markers[i].type = visualization_msgs::msg::Marker::SPHERE;
                    marker_array_msg.markers[i].action = visualization_msgs::msg::Marker::ADD;
                    marker_array_msg.markers[i].pose.position.x = body_positions_solution[i][0];
                    marker_array_msg.markers[i].pose.position.y = body_positions_solution[i][1];
                    marker_array_msg.markers[i].pose.position.z = body_positions_solution[i][2];
                    marker_array_msg.markers[i].pose.orientation.w = 1.0;
                    marker_array_msg.markers[i].scale.x = 0.1;
                    marker_array_msg.markers[i].scale.y = 0.1;
                    marker_array_msg.markers[i].scale.z = 0.1;
                    marker_array_msg.markers[i].color.a = 1.0;
                    marker_array_msg.markers[i].color.r = 0.0;
                    marker_array_msg.markers[i].color.g = 1.0;
                    marker_array_msg.markers[i].color.b = 0.0;

                    marker_array_msg.markers[i + body_positions_solution.size()].header = msg->header;
                    marker_array_msg.markers[i + body_positions_solution.size()].ns = bodies[i].name + "_desired";
                    marker_array_msg.markers[i + body_positions_solution.size()].id = pt_idx;
                    marker_array_msg.markers[i + body_positions_solution.size()].type = visualization_msgs::msg::Marker::SPHERE;
                    marker_array_msg.markers[i + body_positions_solution.size()].action = visualization_msgs::msg::Marker::ADD;
                    marker_array_msg.markers[i + body_positions_solution.size()].pose.position.x = pt.transforms[i].translation.x;
                    marker_array_msg.markers[i + body_positions_solution.size()].pose.position.y = pt.transforms[i].translation.y;
                    marker_array_msg.markers[i + body_positions_solution.size()].pose.position.z = pt.transforms[i].translation.z;
                    marker_array_msg.markers[i + body_positions_solution.size()].pose.orientation.w = 1.0;
                    marker_array_msg.markers[i + body_positions_solution.size()].scale.x = 0.1;
                    marker_array_msg.markers[i + body_positions_solution.size()].scale.y = 0.1;
                    marker_array_msg.markers[i + body_positions_solution.size()].scale.z = 0.1;
                    marker_array_msg.markers[i + body_positions_solution.size()].color.a = 1.0;
                    marker_array_msg.markers[i + body_positions_solution.size()].color.r = 1.0;
                    marker_array_msg.markers[i + body_positions_solution.size()].color.g = 0.0;
                    marker_array_msg.markers[i + body_positions_solution.size()].color.b = 0.0;
                }
                markers_pub_->publish(marker_array_msg);
            }
            pt_idx++;
        }
        this->robot_joints_pub_->publish(out_msg);
    }

    void robot_desc_cb(const std_msgs::msg::String::SharedPtr msg)
    {
        std::cout << "Subscribed to robot description" << std::endl;
        robot_.build_model(msg->data.c_str());
    }

private:
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr robot_joints_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_desc_sub_;
    rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr foot_desired_sub_;
    rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr contact_right_sub_;
    rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr contact_left_sub_;

    IKRobot robot_;
    bool foot_right_contact_;
    bool foot_left_contact_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IKNode>());
    rclcpp::shutdown();
    return 0;
}