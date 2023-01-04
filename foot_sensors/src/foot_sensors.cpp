#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/time_reference.hpp"
#include <time.h>

#include <gpiod.hpp>

using namespace std::chrono_literals;


class TimeTrigger : public rclcpp::Node
{
public:
    TimeTrigger()
    : Node("gpio_time_trigger")
    {
        this->declare_parameter<std::string>("gpio_chip");
        this->declare_parameter<int64_t>("gpio_line");
        this->declare_parameter<std::string>("gpio_edge");
        publisher_ = this->create_publisher<sensor_msgs::msg::TimeReference>("~/time", 10);
        gpio_monitor_ = std::thread(&TimeTrigger::gpio_monitor_loop, this);
    }

    ~TimeTrigger()
    {
        gpio_monitor_.join();
    }

private:
    void gpio_monitor_loop()
    {
        auto chip_name = this->get_parameter("gpio_chip").as_string();
        auto gpio_line = this->get_parameter("gpio_line").as_int();
        auto gpio_edge = this->get_parameter("gpio_edge").as_string();
        gpiod::chip chip(chip_name);
        gpiod::line line = chip.get_line(gpio_line);
        if (gpio_edge == "rising") {
            line.request({this->get_name(), gpiod::line_request::EVENT_RISING_EDGE, 0});
        } else if (gpio_edge == "falling") {
            line.request({this->get_name(), gpiod::line_request::EVENT_FALLING_EDGE, 0});
        } else if (gpio_edge == "both") {
            line.request({this->get_name(), gpiod::line_request::EVENT_BOTH_EDGES, 0});
        } else {
             RCLCPP_ERROR_STREAM(this->get_logger(), "invalid edge type: " << gpio_edge);
        }

        while (rclcpp::ok()) {
            bool has_event = line.event_wait(500ms);
            if (has_event) {
                gpiod::line_event event = line.event_read();
                auto stamp_now_ros = this->now();
                // get current monotonic time to translate timestamp
                struct timespec stamp_now_monotonic;
                clock_gettime(CLOCK_MONOTONIC, &stamp_now_monotonic);
                uint64_t stamp_now_monotonic_ns = stamp_now_monotonic.tv_sec*1000000000 + stamp_now_monotonic.tv_nsec;
                uint64_t time_ros_ref_ns = stamp_now_ros.nanoseconds() - stamp_now_monotonic_ns;
                uint64_t event_time_ns = event.timestamp.count();
                uint64_t event_time_ros_ns = event_time_ns + time_ros_ref_ns;
                auto message = sensor_msgs::msg::TimeReference();
                message.header.stamp = stamp_now_ros;
                message.time_ref.sec = event_time_ros_ns  / 1000000000;
                message.time_ref.nanosec = event_time_ros_ns  % 1000000000;
                if (event.event_type == gpiod::line_event::RISING_EDGE) {
                    message.source = "rising";
                } else {
                    message.source = "falling";
                }
                publisher_->publish(message);
            }
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::TimeReference>::SharedPtr publisher_;
    std::thread gpio_monitor_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TimeTrigger>());
    rclcpp::shutdown();
    return 0;
}
