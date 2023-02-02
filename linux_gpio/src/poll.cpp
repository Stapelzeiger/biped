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
        this->declare_parameter<bool>("gpio_active_low", false);

        init_gpio();

        double dt = 1 / this->declare_parameter<double>("update_rate", 100);

        publisher_ = this->create_publisher<biped_bringup::msg::StampedBool>("~/gpio", 10);
        timer_ = this->create_wall_timer(1s * dt, std::bind(&Poll::timer_callback, this));

    }

    void init_gpio()
    {
        auto chip_name = this->get_parameter("gpio_chip").as_string();
        auto gpio_line = this->get_parameter("gpio_line").as_int();

        chip_.open(chip_name);
        line_ = chip_.get_line(gpio_line);
        line_.request({this->get_name(), gpiod::line_request::DIRECTION_INPUT, 0});
    }
private:

    void timer_callback()
    {
        
        auto message = biped_bringup::msg::StampedBool();
        message.header.stamp = this->now();
        message.data = line_.get_value() ^ this->get_parameter("gpio_active_low").as_bool();
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
