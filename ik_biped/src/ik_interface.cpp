#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "ik_class_pin.hpp"

using namespace std::placeholders;

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber()
        : Node("ik_subscriber_urdf")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/robot_description", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
        IKRobot ik_robot;
        ik_robot.build_model(msg->data.c_str());
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    const std::string urdf_filename = std::string("../../biped_robot_description/urdf/custom_robot.urdf");

    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}