/*
    Takes the sensor_msgs/TimeReference from /vectornav/time_syncin
    published by the vectornav driver and creates a sensor_msgs/TimeReference
    message that is published once for every sync_in event with the correct 
    time of the sync_in event.
*/
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/time_reference.hpp"

using namespace std::chrono_literals;


class VNSyncInEvent : public rclcpp::Node
{
public:
    VNSyncInEvent()
    : Node("vectornav_syncin_event")
    {
        this->declare_parameter<double>("sync_tol", 0.01);
        publisher_ = this->create_publisher<sensor_msgs::msg::TimeReference>("~/trigger_time", 10);
        syncin_sub_ = this->create_subscription<sensor_msgs::msg::TimeReference>(
            "~/vectronav_syncin_ref", 10, std::bind(&VNSyncInEvent::sync_ref_cb, this, std::placeholders::_1));
    }

private:

    void sync_ref_cb(const sensor_msgs::msg::TimeReference::SharedPtr msg)
    {
        rclcpp::Time header_time(msg->header.stamp);
        rclcpp::Duration syncin_age(msg->time_ref.sec, msg->time_ref.nanosec);
        rclcpp::Time syncin_time = header_time - syncin_age;
        double sync_tol = this->get_parameter("sync_tol").as_double();
        if (last_syncin_time.nanoseconds() == 0 || fabs((syncin_time - last_syncin_time).seconds()) > sync_tol) {
            last_syncin_time = syncin_time;
            auto out = sensor_msgs::msg::TimeReference();
            out.header = msg->header;
            out.source = msg->source;
            out.time_ref = syncin_time;
            publisher_->publish(out);
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::TimeReference>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::TimeReference>::SharedPtr syncin_sub_;
    rclcpp::Time last_syncin_time;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VNSyncInEvent>());
    rclcpp::shutdown();
    return 0;
}
