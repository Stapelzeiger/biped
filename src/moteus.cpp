#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

#include <stdexcept>
#include <tuple>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "mjbots/moteus/moteus_protocol.h"
#include "mjbots/moteus/pi3hat_moteus_interface.h"
#pragma GCC diagnostic pop

using namespace mjbots;
using namespace std::chrono_literals;


class MoteusServo: public rclcpp::Node
{
public:
    MoteusServo(const rclcpp::NodeOptions & options) : Node("MoteusServo", options)
    {
        RCLCPP_INFO(this->get_logger(), "Moteus Servo Driver");
        // this->declare_parameter<int64_t>("test", 1);
        // RCLCPP_INFO_STREAM(this->get_logger(), "init value test: " << this->get_parameter("test").as_int());

        // Joint parameters
        this->declare_parameter<std::vector<std::string>>("joints");
        int64_t can_id = 1;
        joint_names_ = this->get_parameter("joints").as_string_array();
        for (auto j : joint_names_)
        {
            this->declare_parameter<int64_t>(j+"/can_id", can_id++);
            this->declare_parameter<int64_t>(j+"/can_bus", 1);
        }

        // Moteus buffers
        moteus_command_buf_.clear();
        for (auto j : joint_names_) {
            int id = this->get_parameter(j+"/can_id").as_int();
            int bus = this->get_parameter(j+"/can_bus").as_int();
            moteus_command_buf_.push_back({});
            moteus_command_buf_.back().id = id;
            moteus_command_buf_.back().bus = bus;
            joint_uid_to_name_[servo_uid(bus, id)] = j;
            std::cout << "Joint: " << j << " bus: " << bus << " id: " << id << std::endl;
        }
        moteus_reply_buf_.resize(moteus_command_buf_.size() * 2); // larger in case there's addition messages

        moteus::Pi3HatMoteusInterface::Options interface_options;
        interface_options.cpu = 1; // TODO make param
        moteus_interface_ = std::make_shared<moteus::Pi3HatMoteusInterface>(interface_options);

        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("~/joint_states", 10);
        joint_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            "~/joint_traj", 10, std::bind(&MoteusServo::traj_cb, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Moteus Servo ready");


        // const int r = ::mlockall(MCL_CURRENT | MCL_FUTURE);
        // if (r < 0) {
        //     throw std::runtime_error("Error locking memory");
        // }
        timer_ = this->create_wall_timer(10ms, std::bind(&MoteusServo::timer_cb, this));
    }

private:
    
    void timer_cb()
    {
        // RCLCPP_INFO(this->get_logger(), "timer cb");

        // TODO fill command_buf with setpoints
        moteus::PositionResolution cmd_res;
        cmd_res.position = moteus::Resolution::kInt16;
        cmd_res.velocity = moteus::Resolution::kInt16;
        cmd_res.feedforward_torque = moteus::Resolution::kInt16;
        cmd_res.kp_scale = moteus::Resolution::kIgnore;
        cmd_res.kd_scale = moteus::Resolution::kIgnore;
        cmd_res.maximum_torque = moteus::Resolution::kIgnore;
        cmd_res.stop_position = moteus::Resolution::kIgnore;
        cmd_res.watchdog_timeout = moteus::Resolution::kIgnore;
        moteus::QueryCommand query; // default query
        assert(query.any_set());
        for (size_t joint_idx = 0; joint_idx < joint_names_.size(); joint_idx++)
        {
            // bus & id set in constructor
            moteus_command_buf_[joint_idx].resolution = cmd_res;
            moteus_command_buf_[joint_idx].mode = moteus::Mode::kStopped;
            moteus_command_buf_[joint_idx].position = moteus::PositionCommand();
            moteus_command_buf_[joint_idx].position.position = 0;
            moteus_command_buf_[joint_idx].position.velocity = 0;
            moteus_command_buf_[joint_idx].position.feedforward_torque = 0;
            moteus_command_buf_[joint_idx].query = query;

            if (joint_traj_msg_) // TODO timeout
            {
                auto pos = std::find(joint_traj_msg_->joint_names.begin(), joint_traj_msg_->joint_names.end(), joint_names_[joint_idx]);
                if (pos != joint_traj_msg_->joint_names.end())
                {
                    moteus_command_buf_[joint_idx].mode = moteus::Mode::kPosition;
                    moteus_command_buf_[joint_idx].position.position = joint_traj_msg_->points[0].positions[std::distance(joint_traj_msg_->joint_names.begin(), pos)];
                }
            }
        }

        moteus::Pi3HatMoteusInterface::Data moteus_io_data;
        moteus_io_data.commands = { moteus_command_buf_.data(), moteus_command_buf_.size() };
        moteus_io_data.replies = { moteus_reply_buf_.data(), moteus_reply_buf_.size() };
        auto promise = std::make_shared<std::promise<moteus::Pi3HatMoteusInterface::Output>>();
        moteus_interface_->Cycle(
            moteus_io_data,
            [promise](const moteus::Pi3HatMoteusInterface::Output& output) {
                promise->set_value(output);
            });
        std::future<moteus::Pi3HatMoteusInterface::Output> can_result = promise->get_future();
        assert(can_result.valid());
        const auto current_values = can_result.get(); // waits for result
        const auto rx_count = current_values.query_result_size;
        std::cout << "rx count " << rx_count << std::endl;
        std::map<std::string, moteus::QueryResult> values;
        for (size_t i = 0; i < rx_count; i++)
        {
            int bus = moteus_reply_buf_[i].bus;
            int id = moteus_reply_buf_[i].id;
            const std::string& joint_name = joint_uid_to_name_[servo_uid(bus, id)];
            float pos = moteus_reply_buf_[i].result.position;
            int fault = moteus_reply_buf_[i].result.fault;
            std::cout << joint_name << " : " << fault << std::endl;
            values[joint_name] = moteus_reply_buf_[i].result;
        }
        // publish joint states
        sensor_msgs::msg::JointState msg;
        msg.header.stamp = this->now();
        for (auto joint : joint_names_)
        {
            msg.name.push_back(joint);
            msg.position.push_back(values[joint].position);
            msg.velocity.push_back(values[joint].velocity);
            msg.effort.push_back(values[joint].torque);
        }
        joint_pub_->publish(msg);
    }

    void traj_cb(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
    {
        joint_traj_msg_ = msg;
    }

    static int servo_uid(int bus, int id)
    {
        return (bus << 8) + id;
    }

    std::vector<std::string> joint_names_;
    std::map<int, std::string> joint_uid_to_name_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_sub_;
    trajectory_msgs::msg::JointTrajectory::SharedPtr joint_traj_msg_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<moteus::Pi3HatMoteusInterface::ServoCommand> moteus_command_buf_;
    std::vector<moteus::Pi3HatMoteusInterface::ServoReply> moteus_reply_buf_;
    std::shared_ptr<moteus::Pi3HatMoteusInterface> moteus_interface_;
};


RCLCPP_COMPONENTS_REGISTER_NODE(MoteusServo)
