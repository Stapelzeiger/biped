#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include "ik_class_pin.hpp"

using namespace std::placeholders;

class IKNode : public rclcpp::Node
{

public:
    IKNode() : Node("ik_node")
    {
        robot_desc_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/robot_description", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(), std::bind(&IKNode::robot_desc_cb, this, _1));

        // subscribe to current q
        // subscribe to foot desired

        robot_joints_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&IKNode::timer_pub_robot_joints, this));
    }

private:
    void timer_pub_robot_joints()
    {

        Eigen::VectorXd q_current(17, 1);
        q_current << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        Eigen::Vector3d pos_foot_des(3, 1);
        std::cout << q_current << std::endl;
        pos_foot_des << -0.2, 0.0, 0.0;
        double yaw_angle_des = 0.0;
        std::string joint_name = "FL_ANKLE";
        Eigen::VectorXd q_des;
        q_des = robot_.get_desired_q(q_current, pos_foot_des, yaw_angle_des, joint_name);
        std::cout << q_des << std::endl;
        const sensor_msgs::msg::JointState message;

        robot_joints_pub_->publish(message);
    }

    void robot_desc_cb(const std_msgs::msg::String::SharedPtr msg)
    {

        robot_.build_model(msg->data.c_str());
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr robot_joints_pub_;
    size_t count_;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_desc_sub_;
    IKRobot robot_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IKNode>());
    rclcpp::shutdown();
    return 0;
}