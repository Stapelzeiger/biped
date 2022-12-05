/*
    Takes the sensor_msgs/Image and a sensor_msgs/TimeReference
    and republishes the image with the timestamp in the reference
*/
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/time_reference.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

using namespace std::chrono_literals;


class ImgSync : public rclcpp::Node
{
public:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::TimeReference> SyncPolicy;
    ImgSync()
    : Node("sync_image")
    {
        // this->declare_parameter<double>("sync_tol", 0.01);
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("~/image_out", 10);
        image_sub_.subscribe(this, "~/image");
        time_sub_.subscribe(this, "~/trigger");
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(3), image_sub_, time_sub_);
        sync_->registerCallback(&ImgSync::img_cb, this);
    }

private:
    void img_cb(const sensor_msgs::msg::Image::SharedPtr img, const sensor_msgs::msg::TimeReference::SharedPtr time_ref)
    {
        sensor_msgs::msg::Image out = *img;
        out.header.stamp = time_ref->time_ref;
        publisher_->publish(out);
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::TimeReference> time_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImgSync>());
    rclcpp::shutdown();
    return 0;
}
