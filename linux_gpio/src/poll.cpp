#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "linux_gpio/msg/stamped_bool.hpp"

#include <time.h>

#include <gpiod.hpp>

using namespace std::chrono_literals;

class Poll : public rclcpp::Node
{
public:
    Poll()
    : Node("gpio_poll")
    {
        this->declare_parameter<std::string>("gpio_chip");
        this->declare_parameter<int64_t>("gpio_line");
        this->declare_parameter<std::string>("gpio_edge");

        init_gpio();

        publisher_ = this->create_publisher<linux_gpio::msg::StampedBool>("~/gpio", 10);
        timer_ = this->create_wall_timer(100ms, std::bind(&Poll::timer_callback, this));

    }

    void init_gpio()
    {
        auto chip_name = this->get_parameter("gpio_chip").as_string();
        auto gpio_line = this->get_parameter("gpio_line").as_int();
        auto gpio_edge = this->get_parameter("gpio_edge").as_string();

        std::cout << "chip_name: " << chip_name << std::endl;
        std::cout << "gpio_line: " << gpio_line << std::endl;
        std::cout << "gpio_edge: " << gpio_edge << std::endl;

        chip.open(chip_name);

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
    }
private:

    void timer_callback()
    {
        
        auto message = linux_gpio::msg::StampedBool();
        // message.header.stamp = this->now();
        // message.data = line.get_value();
        // RCLCPP_INFO(this->get_logger(), "Publishing: '%d'", message.data);
        // publisher_->publish(message);

    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<linux_gpio::msg::StampedBool>::SharedPtr publisher_;
    gpiod::chip chip;
    gpiod::line line;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Poll>());
    rclcpp::shutdown();
    return 0;
}
