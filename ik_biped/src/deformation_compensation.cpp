#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "eigen3/Eigen/Dense"
#include <math.h>

using namespace std::placeholders;
using namespace std::chrono_literals;
class DeformationCompensationNode : public rclcpp::Node
{

public:
    DeformationCompensationNode() : Node("def_comp_node")
    {

        measured_joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10, std::bind(&DeformationCompensationNode::actual_joint_states_cb, this, _1));

        joint_traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            "joint_trajectory", 10, std::bind(&DeformationCompensationNode::joint_traj_cb, this, _1));

        compensated_joint_states_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states_compensated", 1);
        compensated_joint_trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("joint_trajectory_compensated", 1);

        // Parameters.
        double k_L_HAA = this->declare_parameter("k_L_HAA", 0.0);
        double k_L_HFE = this->declare_parameter("k_L_HFE", 0.0);
        double K_L_KFE = this->declare_parameter("k_L_KFE", 0.0);
        double K_R_HAA = this->declare_parameter("k_R_HAA", 0.0);
        double K_R_HFE = this->declare_parameter("k_R_HFE", 0.0);
        double K_R_KFE = this->declare_parameter("k_R_KFE", 0.0);
        compensation_dead_band_torque_ = this->declare_parameter<std::vector<double>>("compensation_dead_band_torque", {0.0, 0.0, 0.0});
        if (compensation_dead_band_torque_.size() != 3)
        {
            throw std::runtime_error("compensation_dead_band_torque must have 3 elements");
        }
        torque_lp_filter_tau_ = this->declare_parameter("torque_lp_filter_tau", 0.01);
        last_update_time_ = this->now();

        K_right_leg_ = Eigen::Vector3d(K_R_HAA, K_R_HFE, K_R_KFE).asDiagonal();
        K_left_leg_ = Eigen::Vector3d(k_L_HAA, k_L_HFE, K_L_KFE).asDiagonal();
    }

private:
    void actual_joint_states_cb(sensor_msgs::msg::JointState::ConstSharedPtr msg)
    {
        sensor_msgs::msg::JointState compensated_joint_states;
        compensated_joint_states.header = msg->header;
        compensated_joint_states.name = msg->name;
        compensated_joint_states.position = msg->position;
        compensated_joint_states.velocity = msg->velocity;
        compensated_joint_states.effort = msg->effort;

        Eigen::Vector3d q_right = Eigen::Vector3d::Zero();
        Eigen::Vector3d q_left = Eigen::Vector3d::Zero();
        Eigen::Vector3d tau_right_in = Eigen::Vector3d::Zero();
        Eigen::Vector3d tau_left_in = Eigen::Vector3d::Zero();

        for (unsigned int js_idx = 0; js_idx < msg->name.size(); js_idx++)
        {
            for (unsigned int vect_idx = 0; vect_idx < 3; vect_idx++)
            {
                if (msg->name[js_idx] == joint_names_right_[vect_idx])
                {
                    if (js_idx < msg->effort.size())
                    {
                        tau_right_in[vect_idx] = msg->effort[js_idx];
                    }
                    if (js_idx < msg->position.size())
                    {
                        q_right[vect_idx] = msg->position[js_idx];
                    }
                }
                if (msg->name[js_idx] == joint_names_left_[vect_idx])
                {
                    if (js_idx < msg->effort.size())
                    {
                        tau_left_in[vect_idx] = msg->effort[js_idx];
                    }
                    if (js_idx < msg->position.size())
                    {
                        q_left[vect_idx] = msg->position[js_idx];
                    }
                }
            }
        }

        for (unsigned int vect_idx = 0; vect_idx < 3; vect_idx++)
        {
            if (std::abs(tau_right_in[vect_idx]) < compensation_dead_band_torque_[vect_idx])
            {
                tau_right_in[vect_idx] = 0.0;
            }
            if (std::abs(tau_left_in[vect_idx]) < compensation_dead_band_torque_[vect_idx])
            {
                tau_left_in[vect_idx] = 0.0;
            }
        }

        // update low pass filter
        rclcpp::Time now = msg->header.stamp;
        double dt = (now - last_update_time_).seconds();
        dt = std::clamp(dt, 0.001, 0.1);
        last_update_time_ = now;
        double alpha = dt / (dt + torque_lp_filter_tau_);
        tau_right_ = alpha * tau_right_in + (1 - alpha) * tau_right_;
        tau_left_ = alpha * tau_left_in + (1 - alpha) * tau_left_;

        Eigen::Vector3d q_right_corrected = q_right + K_right_leg_ * tau_right_;
        Eigen::Vector3d q_left_corrected = q_left + K_left_leg_ * tau_left_;

        for (unsigned int js_idx = 0; js_idx < msg->name.size(); js_idx++)
        {
            for (unsigned int vect_idx = 0; vect_idx < 3; vect_idx++)
            {
                if (msg->name[js_idx] == joint_names_right_[vect_idx])
                {
                    if (js_idx < compensated_joint_states.position.size())
                    {
                        compensated_joint_states.position[js_idx] = q_right_corrected[vect_idx];
                    }
                }
                if (msg->name[js_idx] == joint_names_left_[vect_idx])
                {
                    if (js_idx < compensated_joint_states.position.size())
                    {
                        compensated_joint_states.position[js_idx] = q_left_corrected[vect_idx];
                    }
                }
            }
        }

        compensated_joint_states_pub_->publish(compensated_joint_states);
    }

    void joint_traj_cb(trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg)
    {
        trajectory_msgs::msg::JointTrajectory compensated_joint_trajectory;
        compensated_joint_trajectory.header = msg->header;
        compensated_joint_trajectory.joint_names = msg->joint_names;

        // tau_left_ = Eigen::Vector3d::Zero();
        // tau_right_ = Eigen::Vector3d::Zero();

        for (unsigned int pt_idx = 0; pt_idx < msg->points.size(); pt_idx++)
        {
            trajectory_msgs::msg::JointTrajectoryPoint compensated_pt = msg->points[pt_idx];

            Eigen::Vector3d q_right = Eigen::Vector3d::Zero();
            Eigen::Vector3d q_left = Eigen::Vector3d::Zero();

            for (unsigned int js_idx = 0; js_idx < msg->joint_names.size(); js_idx++)
            {
                for (unsigned int vect_idx = 0; vect_idx < 3; vect_idx++)
                {
                    if (msg->joint_names[js_idx] == joint_names_right_[vect_idx])
                    {
                        if (js_idx < msg->points[pt_idx].positions.size())
                        {
                            q_right[vect_idx] = msg->points[pt_idx].positions[js_idx];
                        }
                    }
                    if (msg->joint_names[js_idx] == joint_names_left_[vect_idx])
                    {
                        if (js_idx < msg->points[pt_idx].positions.size())
                        {
                            q_left[vect_idx] = msg->points[pt_idx].positions[js_idx];
                        }
                    }
                }
            }

            Eigen::Vector3d q_right_corrected = q_right - K_right_leg_ * tau_right_;
            Eigen::Vector3d q_left_corrected = q_left - K_left_leg_ * tau_left_;

            for (unsigned int js_idx = 0; js_idx < msg->joint_names.size(); js_idx++)
            {
                for (unsigned int vect_idx = 0; vect_idx < 3; vect_idx++)
                {
                    if (msg->joint_names[js_idx] == joint_names_right_[vect_idx])
                    {
                        if (js_idx < compensated_pt.positions.size())
                        {
                            compensated_pt.positions[js_idx] = q_right_corrected[vect_idx];
                        }
                    }
                    if (msg->joint_names[js_idx] == joint_names_left_[vect_idx])
                    {
                        if (js_idx < compensated_pt.positions.size())
                        {
                            compensated_pt.positions[js_idx] = q_left_corrected[vect_idx];
                        }
                    }
                }
            }
            compensated_joint_trajectory.points.push_back(compensated_pt);
        }
        compensated_joint_trajectory_pub_->publish(compensated_joint_trajectory);
    }

private:
    Eigen::Matrix3d K_right_leg_;
    Eigen::Matrix3d K_left_leg_;
    Eigen::Vector3d tau_right_;
    Eigen::Vector3d tau_left_;
    double torque_lp_filter_tau_;
    rclcpp::Time last_update_time_;
    std::vector<double> compensation_dead_band_torque_;
    static const std::vector<std::string> joint_names_right_;
    static const std::vector<std::string> joint_names_left_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr measured_joint_states_sub_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_traj_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr compensated_joint_states_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr compensated_joint_trajectory_pub_;
};

const std::vector<std::string> DeformationCompensationNode::joint_names_right_ = {"R_HAA", "R_HFE", "R_KFE"};
const std::vector<std::string> DeformationCompensationNode::joint_names_left_ = {"L_HAA", "L_HFE", "L_KFE"};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DeformationCompensationNode>());
    rclcpp::shutdown();
    return 0;
}