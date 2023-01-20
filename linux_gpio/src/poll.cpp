#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "biped_bringup/msg/stamped_bool.hpp"

#include <time.h>

#include <gpiod.hpp>

using namespace std::chrono_literals;


// https://developer.download.nvidia.com/assets/embedded/secure/jetson/agx_orin/Jetson_AGX_Orin_DevKit_Carrier_Board_Specification_SP-10900-001_v1.0.pdf?buxW4aV1stbg3PkFvvKEzVumg_ZXiCs4usr5pCpE4CkERA0NBpBb6MIouMDh2Wufxdk6qxK1bPArMuYjqqgwA2HRP_ALPR9-jLIcnveQazMslfod-JkidAo0EsZ73yNQ423k7L1wzaqlfayDS6Ou_GqR2yr3Bsdzvr-pRknhW4JXk2-MCJBwfoSyXAn5wV9FJUBhBwVvCTga0MAQqlH4BA9gIKVqgZ7TRfLGZsxp&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9

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

        publisher_ = this->create_publisher<biped_bringup::msg::StampedBool>("~/gpio", 10);
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

        chip_.open(chip_name);

        line_ = chip_.get_line(gpio_line);
        if (gpio_edge == "rising") {
            line_.request({this->get_name(), gpiod::line_request::EVENT_RISING_EDGE, 0});
        } else if (gpio_edge == "falling") {
            line_.request({this->get_name(), gpiod::line_request::EVENT_FALLING_EDGE, 0});
        } else if (gpio_edge == "both") {
            line_.request({this->get_name(), gpiod::line_request::EVENT_BOTH_EDGES, 0});
        } else {
                RCLCPP_ERROR_STREAM(this->get_logger(), "invalid edge type: " << gpio_edge);
        }
    }
private:

    void timer_callback()
    {
        
        auto message = biped_bringup::msg::StampedBool();
        message.header.stamp = this->now();
        message.data = line_.get_value();
        RCLCPP_INFO(this->get_logger(), "Publishing: '%d'", message.data);
        publisher_->publish(message);

    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<biped_bringup::msg::StampedBool>::SharedPtr publisher_;
    gpiod::chip chip_;
    gpiod::line line_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Poll>());
    rclcpp::shutdown();
    return 0;
}
