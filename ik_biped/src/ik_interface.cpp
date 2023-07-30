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
#include "std_msgs/msg/float64_multi_array.hpp"
#include "ik_class_pin.hpp"
#include "biped_bringup/msg/stamped_bool.hpp"
#include <math.h>

using namespace std::placeholders;

class IKNode : public rclcpp::Node
{

public:
    IKNode() : Node("ik_node")
    {
        robot_desc_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/robot_description", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(), std::bind(&IKNode::robot_desc_cb, this, _1));

        body_desired_sub_ = this->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
            "body_trajectories", 10, std::bind(&IKNode::body_desired_cb, this, _1));

        measured_joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&IKNode::actual_joint_states_cb, this, _1));

        odom_baselink_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", 10, std::bind(&IKNode::odom_baselink_cb, this, _1));

        robot_joints_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("joint_trajectory", 10);
        markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/markers", 10);

        joint_states_for_EL_eq_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states_for_EL_eq", 10);

        gravity_torque_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("gravity_torque", 10);
        corriolis_torque_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("corriolis_torque", 10);
        inertia_torque_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("inertia_torque", 10);

        acc_foot_computed_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("acc_foot_computed", 10);

    }

private:
    void contact_cb(biped_bringup::msg::StampedBool::SharedPtr msg, const std::string &joint_name)
    {
        contact_states_[joint_name] = msg;
    }

    void actual_joint_states_cb(sensor_msgs::msg::JointState::SharedPtr msg)
    {
        encoder_joint_states_.clear();
        for (size_t i = 0; i < msg->name.size(); i++) {
            const auto &name = msg->name[i];
            const auto &pos = msg->position[i];
            const auto &vel = msg->velocity[i];
            const auto &eff = msg->effort[i];
            encoder_joint_states_.push_back(IKRobot::JointState{name, pos, vel, eff});
        }
        time_encoder_joint_state_ = rclcpp::Time(msg->header.stamp);
    }

    void odom_baselink_cb(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        odom_baselink_.position = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        odom_baselink_.orientation = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        odom_baselink_.linear_velocity = Eigen::Vector3d(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
        odom_baselink_.angular_velocity = Eigen::Vector3d(msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);
        time_odom_baselink_ = rclcpp::Time(msg->header.stamp);
    }

    void body_desired_cb(trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg)
    {
        if (!robot_.has_model()) {
            RCLCPP_ERROR_SKIPFIRST_THROTTLE(this->get_logger(), *this->get_clock(), 1000 /* [ms] */, "No robot model loaded");
            return;
        }
        // ensure there are contact subscribers
        for (const auto &name : msg->joint_names) {
            if (contact_subs_.find(name) == contact_subs_.end())
            {
                contact_subs_[name] = this->create_subscription<biped_bringup::msg::StampedBool>(
                    "~/contact_" + name, 1, [this, name](const biped_bringup::msg::StampedBool::SharedPtr msg) {contact_cb(msg, name);});
            }
        }
        // check time of contact data
        for (const auto &contact : contact_states_)
        {
            auto now = rclcpp::Time(msg->header.stamp);
            double dt = (now - rclcpp::Time(contact.second->header.stamp)).seconds();
            if (fabs(dt) > 0.1)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100 /* [ms] */, "contact %s sensor data is too old: %f", contact.first.c_str(), dt);
            }
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
            // check if baselink is missing
            if (std::find(msg->joint_names.begin(), msg->joint_names.end(), "base_link") == msg->joint_names.end()) {
                RCLCPP_ERROR(this->get_logger(), "base_link missing in MultiDofJoints");
                return;
            }

            for (size_t i = 0; i < msg->joint_names.size(); i++) {
                const auto &name = msg->joint_names[i];
                const auto &pos = Eigen::Vector3d(pt.transforms[i].translation.x, pt.transforms[i].translation.y, pt.transforms[i].translation.z);
                const auto &rot = Eigen::Quaterniond(pt.transforms[i].rotation.w, pt.transforms[i].rotation.x, pt.transforms[i].rotation.y, pt.transforms[i].rotation.z);
                IKRobot::BodyState body_state(name, pos, rot);
                if (pt.velocities.size() > 0) {
                    body_state.linear_velocity = Eigen::Vector3d(pt.velocities[i].linear.x, pt.velocities[i].linear.y, pt.velocities[i].linear.z);
                }
                if (pt.accelerations.size() > 0) {
                    body_state.linear_acceleration = Eigen::Vector3d(pt.accelerations[i].linear.x, pt.accelerations[i].linear.y, pt.accelerations[i].linear.z);
                }
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

                if (contact_states_.find(name) != contact_states_.end()) {
                    body_state.in_contact = contact_states_[name]->data;
                } else {
                    body_state.in_contact = false;
                }

                bodies.push_back(body_state);
            }
            RCLCPP_DEBUG_STREAM(this->get_logger(), "solving with bodies :");
            for (const auto &body : bodies) {
                RCLCPP_DEBUG_STREAM(this->get_logger(), "   " << body.name);
            }
            if (fabs((time_odom_baselink_ - time_encoder_joint_state_).seconds()) > 0.1) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100 /* [ms] */, "odom and joint_states are out of sync: ");
            }
            std::vector<Eigen::Vector3d> body_positions_solution;
            std::vector<IKRobot::JointState> joint_states_for_EL_eq;
            Eigen::VectorXd gravity_torque;
            Eigen::VectorXd coriolis_torque;
            Eigen::VectorXd inertia_torque;
            Eigen::VectorXd a_foot_computed;

            std::vector<IKRobot::JointState> joint_states = this->robot_.solve(bodies,
                                                                                odom_baselink_,
                                                                                encoder_joint_states_,
                                                                                body_positions_solution,
                                                                                joint_states_for_EL_eq,
                                                                                gravity_torque,
                                                                                coriolis_torque,
                                                                                inertia_torque,
                                                                                a_foot_computed);
            // publish joint joint_states_for_EL_eq
            sensor_msgs::msg::JointState joint_states_for_EL_eq_msg;
            joint_states_for_EL_eq_msg.header = msg->header;
            joint_states_for_EL_eq_msg.name.resize(joint_states_for_EL_eq.size());
            joint_states_for_EL_eq_msg.position.resize(joint_states_for_EL_eq.size());
            joint_states_for_EL_eq_msg.velocity.resize(joint_states_for_EL_eq.size());
            joint_states_for_EL_eq_msg.effort.resize(joint_states_for_EL_eq.size());

            for (size_t i = 0; i < joint_states_for_EL_eq.size(); i++) {
                joint_states_for_EL_eq_msg.name[i] = joint_states_for_EL_eq[i].name;
                joint_states_for_EL_eq_msg.position[i] = joint_states_for_EL_eq[i].position;
                joint_states_for_EL_eq_msg.velocity[i] = joint_states_for_EL_eq[i].velocity;
                joint_states_for_EL_eq_msg.effort[i] = joint_states_for_EL_eq[i].acceleration;
            }

            joint_states_for_EL_eq_pub_->publish(joint_states_for_EL_eq_msg);

            //pub gravity torque
            std_msgs::msg::Float64MultiArray gravity_torque_msg;
            gravity_torque_msg.data.resize(gravity_torque.size());
            for (size_t i = 0; i < gravity_torque.size(); i++) {
                gravity_torque_msg.data[i] = gravity_torque[i];
            }
            gravity_torque_pub_->publish(gravity_torque_msg);
            // pub coriolis torque
            std_msgs::msg::Float64MultiArray coriolis_torque_msg;
            coriolis_torque_msg.data.resize(coriolis_torque.size());
            for (size_t i = 0; i < coriolis_torque.size(); i++) {
                coriolis_torque_msg.data[i] = coriolis_torque[i];
            }
            corriolis_torque_pub_->publish(coriolis_torque_msg);
            // pub inertia torque
            std_msgs::msg::Float64MultiArray inertia_torque_msg;
            inertia_torque_msg.data.resize(inertia_torque.size());
            for (size_t i = 0; i < inertia_torque.size(); i++) {
                inertia_torque_msg.data[i] = inertia_torque[i];
            }
            inertia_torque_pub_->publish(inertia_torque_msg);

            // pub a_foot_computed
            std_msgs::msg::Float64MultiArray a_foot_computed_msg;
            a_foot_computed_msg.data.resize(a_foot_computed.size());
            for (size_t i = 0; i < a_foot_computed.size(); i++) {
                a_foot_computed_msg.data[i] = a_foot_computed[i];
            }
            acc_foot_computed_pub_->publish(a_foot_computed_msg);

            out_msg.joint_names.resize(joint_states.size());
            for (size_t i = 0; i < joint_states.size(); i++) {
                out_msg.joint_names[i] = joint_states[i].name;
            }
            trajectory_msgs::msg::JointTrajectoryPoint out_pt;

            out_pt.positions.resize(joint_states.size());
            for (size_t i = 0; i < joint_states.size(); i++) {
                out_pt.positions[i] = joint_states[i].position;
            }

            out_pt.velocities.resize(joint_states.size());
            for (size_t i = 0; i < joint_states.size(); i++) {
                out_pt.velocities[i] = joint_states[i].velocity;
            }

            out_pt.effort.resize(joint_states.size());
            for (size_t i = 0; i < joint_states.size(); i++) {
                out_pt.effort[i] = joint_states[i].effort;
            }

            out_pt.time_from_start = pt.time_from_start;
            out_msg.points.push_back(out_pt);

            if (markers_pub_->get_subscription_count() > 0) {
                visualization_msgs::msg::MarkerArray marker_array_msg;
                marker_array_msg.markers.resize(body_positions_solution.size() * 3);
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
                    marker_array_msg.markers[i].scale.x = 0.05;
                    marker_array_msg.markers[i].scale.y = 0.05;
                    marker_array_msg.markers[i].scale.z = 0.05;
                    marker_array_msg.markers[i].color.a = 1.0;
                    marker_array_msg.markers[i].color.r = 0.0;
                    marker_array_msg.markers[i].color.g = 1.0; // green blob
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
                    marker_array_msg.markers[i + body_positions_solution.size()].scale.x = 0.05;
                    marker_array_msg.markers[i + body_positions_solution.size()].scale.y = 0.05;
                    marker_array_msg.markers[i + body_positions_solution.size()].scale.z = 0.05;
                    marker_array_msg.markers[i + body_positions_solution.size()].color.a = 1.0;
                    marker_array_msg.markers[i + body_positions_solution.size()].color.r = 1.0; // red blob
                    marker_array_msg.markers[i + body_positions_solution.size()].color.g = 0.0;
                    marker_array_msg.markers[i + body_positions_solution.size()].color.b = 0.0;


                    marker_array_msg.markers[i + 2*body_positions_solution.size()].header = msg->header;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].ns = bodies[i].name + "_input";
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].id = pt_idx;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].type = visualization_msgs::msg::Marker::SPHERE;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].action = visualization_msgs::msg::Marker::ADD;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].pose.position.x = bodies[i].position[0];
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].pose.position.y = bodies[i].position[1];
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].pose.position.z = bodies[i].position[2];
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].pose.orientation.w = 1.0;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].scale.x = 0.05;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].scale.y = 0.05;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].scale.z = 0.05;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].color.a = 1.0;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].color.r = 0.0;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].color.g = 0.0;
                    marker_array_msg.markers[i + 2*body_positions_solution.size()].color.b = 1.0; // blue blob

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
    rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr body_desired_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr measured_joint_states_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_for_EL_eq_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_baselink_sub_;
    std::map<std::string, rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr> contact_subs_;
    std::map<std::string, biped_bringup::msg::StampedBool::SharedPtr> contact_states_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gravity_torque_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr corriolis_torque_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr inertia_torque_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr acc_foot_computed_pub_;

    IKRobot robot_;
    rclcpp::Time time_encoder_joint_state_;
    std::vector<IKRobot::JointState> encoder_joint_states_;
    IKRobot::BodyState odom_baselink_;
    rclcpp::Time time_odom_baselink_;

};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IKNode>());
    rclcpp::shutdown();
    return 0;
}