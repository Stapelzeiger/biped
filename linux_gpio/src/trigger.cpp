#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/time_reference.hpp"

#include <gpiod.h>

using namespace std::chrono_literals;


class TimeTrigger : public rclcpp::Node
{
public:
    TimeTrigger()
    : Node("gpio_time_trigger")
    {
        this->declare_parameter<double>("frequency", 10);
        double frequency = this->get_parameter("frequency").as_double();
        this->declare_parameter<std::string>("gpio_chip");
        this->declare_parameter<int64_t>("gpio_line");
        this->declare_parameter<double>("pulse_duration", 0.01);
        auto chip_name = this->get_parameter("gpio_chip").as_string();
        auto gpio_line = this->get_parameter("gpio_line").as_int();
        output_chip_ = gpiod_chip_open_lookup(chip_name.c_str());
        if (output_chip_ == NULL) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "could not open gpio-chip: " << chip_name);
            throw std::runtime_error("could not open gpio-chip");
        }
        output_line_ = gpiod_chip_get_line(output_chip_, gpio_line);
        if (output_line_ == NULL) {
            gpiod_chip_close(output_chip_);
            RCLCPP_ERROR_STREAM(this->get_logger(), "could not open gpio: " << gpio_line);
            throw std::runtime_error("could not open gpio");
        }
        int ret = gpiod_line_request_output(output_line_, this->get_name(), 0);
        if (ret < 0) {
            gpiod_line_release(output_line_);
            gpiod_chip_close(output_chip_);
            RCLCPP_ERROR_STREAM(this->get_logger(), "request gpio as output failed");
            throw std::runtime_error("could not set gpio as output");
        }

        publisher_ = this->create_publisher<sensor_msgs::msg::TimeReference>("~/trigger_time", 10);
        timer_ = this->create_wall_timer(
        1s / frequency, std::bind(&TimeTrigger::timer_callback, this));
    }

    ~TimeTrigger()
    {
        gpiod_line_release(output_line_);
        gpiod_chip_close(output_chip_);
    }

private:
    void timer_callback()
    {
        auto message = sensor_msgs::msg::TimeReference();
        message.header.stamp = this->now();
        int ret = gpiod_line_set_value(output_line_, 1);
        if (ret < 0) {
             RCLCPP_ERROR_STREAM(this->get_logger(), "GPIO set failed");
        }

        message.time_ref = message.header.stamp;
        publisher_->publish(message);

        std::this_thread::sleep_for(this->get_parameter("pulse_duration").as_double() * 1s);
        ret = gpiod_line_set_value(output_line_, 0);
        if (ret < 0) {
             RCLCPP_ERROR_STREAM(this->get_logger(), "GPIO set failed");
        }
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::TimeReference>::SharedPtr publisher_;
    struct gpiod_chip *output_chip_;
    struct gpiod_line *output_line_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TimeTrigger>());
    rclcpp::shutdown();
    return 0;
}
