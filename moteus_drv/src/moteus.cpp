#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/temperature.hpp"
#include "std_msgs/msg/float32.hpp"
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

        // Joint parameters
        this->declare_parameter<std::vector<std::string>>("joints");
        int64_t can_id = 1;
        joint_names_ = this->get_parameter("joints").as_string_array();
        nb_joints_ = joint_names_.size();
        joint_state_timeout_ = this->declare_parameter<double>("joint_state_timeout", 0.03);
        joint_command_timeout_ = this->declare_parameter<double>("joint_command_timeout", 0.1);
        this->declare_parameter<double>("global_max_torque", 1e9);
        joint_traj_.resize(nb_joints_);
        moteus_query_res_.resize(nb_joints_);
        joint_offsets_.resize(nb_joints_);
        joint_signs_.resize(nb_joints_);

        // Moteus buffers
        moteus_command_buf_.clear();
        for (size_t joint_idx = 0; joint_idx < nb_joints_; joint_idx++) {
            auto joint_name = joint_names_[joint_idx];
            joint_offsets_[joint_idx] = this->declare_parameter<double>(joint_name + "/offset", 0);
            bool rev = this->declare_parameter<bool>(joint_name + "/reverse", false);
            if (rev) {
                joint_signs_[joint_idx] = -1;
            } else {
                joint_signs_[joint_idx] = 1;
            }
            this->declare_parameter<double>(joint_name + "/kp_scale", 1);
            this->declare_parameter<double>(joint_name + "/kd_scale", 1);
            this->declare_parameter<double>(joint_name + "/max_torque", 1e9);
            
            int id = this->declare_parameter<int64_t>(joint_name + "/can_id", can_id++);
            int bus = this->declare_parameter<int64_t>(joint_name + "/can_bus", 1);
            moteus_command_buf_.push_back({});
            moteus_command_buf_.back().id = id;
            moteus_command_buf_.back().bus = bus;
            joint_uid_to_joint_index_[servo_uid(bus, id)] = joint_idx;
            RCLCPP_INFO_STREAM(this->get_logger(), "Adding joint: " << joint_name << ", CAN bus: " << bus << ", id " << id);
        }
        moteus_reply_buf_.resize(moteus_command_buf_.size() * 2); // larger in case there's addition messages

        moteus::Pi3HatMoteusInterface::Options interface_options;
        interface_options.cpu = 1; // TODO make param
        moteus_interface_ = std::make_shared<moteus::Pi3HatMoteusInterface>(interface_options);

        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("~/joint_states", 10);
        joint_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            "~/joint_traj", 10, std::bind(&MoteusServo::traj_cb, this, std::placeholders::_1));
        for (auto joint_name : joint_names_) {
            volt_pub_.push_back(this->create_publisher<std_msgs::msg::Float32>("~/" + joint_name + "/voltage", 10));
            temp_pub_.push_back(this->create_publisher<sensor_msgs::msg::Temperature>("~/" + joint_name + "/temperature", 10));
        }

        // const int r = ::mlockall(MCL_CURRENT | MCL_FUTURE);
        // if (r < 0) {
        //     throw std::runtime_error("Error locking memory");
        // }
        this->declare_parameter<double>("update_rate");
        double update_rate =  this->get_parameter("update_rate").as_double();
        timer_ = this->create_wall_timer(1000ms/update_rate, std::bind(&MoteusServo::timer_cb, this));
    }

private:

    void timer_cb()
    {
        auto now = this->now();
        moteus::PositionResolution cmd_res;
        cmd_res.position = moteus::Resolution::kInt16;
        cmd_res.velocity = moteus::Resolution::kInt16;
        cmd_res.feedforward_torque = moteus::Resolution::kInt16;
        cmd_res.kp_scale = moteus::Resolution::kInt8;
        cmd_res.kd_scale = moteus::Resolution::kInt8;
        cmd_res.maximum_torque = moteus::Resolution::kInt16;
        cmd_res.stop_position = moteus::Resolution::kIgnore;
        cmd_res.watchdog_timeout = moteus::Resolution::kIgnore;
        moteus::QueryCommand query; // default query
        assert(query.any_set());
        double global_max_torque = this->get_parameter("global_max_torque").as_double();
        for (size_t joint_idx = 0; joint_idx < nb_joints_; joint_idx++)
        {
            // bus & id set in constructor
            moteus_command_buf_[joint_idx].resolution = cmd_res;
            moteus_command_buf_[joint_idx].mode = moteus::Mode::kStopped;
            moteus_command_buf_[joint_idx].position = moteus::PositionCommand();
            moteus_command_buf_[joint_idx].position.position = std::numeric_limits<double>::quiet_NaN();
            moteus_command_buf_[joint_idx].position.velocity = 0;
            moteus_command_buf_[joint_idx].position.feedforward_torque = 0;

            double max_torque = this->get_parameter(joint_names_[joint_idx] + "/max_torque").as_double();
            max_torque = std::min(max_torque, global_max_torque);
            moteus_command_buf_[joint_idx].position.maximum_torque = max_torque;
            moteus_command_buf_[joint_idx].position.kp_scale = this->get_parameter(joint_names_[joint_idx] + "/kp_scale").as_double();
            moteus_command_buf_[joint_idx].position.kd_scale = this->get_parameter(joint_names_[joint_idx] + "/kd_scale").as_double();
            
            moteus_command_buf_[joint_idx].query = query;

            if (joint_traj_[joint_idx].points.size() > 0) {
                // check for timeout and index in trajectory
                if ((now - joint_traj_[joint_idx].header.stamp).seconds() > joint_command_timeout_) {
                    joint_traj_[joint_idx].points.clear();
                    RCLCPP_WARN_STREAM(this->get_logger(), "Joint " << joint_names_[joint_idx] << " timed out");
                } else {
                    const double revolutions = 1/(2*M_PI);
                    size_t traj_idx = 0; // TODO: handle multiple traj points
                    moteus_command_buf_[joint_idx].mode = moteus::Mode::kPosition;
                    if (joint_traj_[joint_idx].points[traj_idx].positions.size() == 1) {
                        double p = joint_traj_[joint_idx].points[traj_idx].positions[0];
                        moteus_command_buf_[joint_idx].position.position = revolutions * (p * joint_signs_[joint_idx] + joint_offsets_[joint_idx]);
                    }
                    if (joint_traj_[joint_idx].points[traj_idx].velocities.size() == 1) {
                        double v = joint_traj_[joint_idx].points[traj_idx].velocities[0];
                        moteus_command_buf_[joint_idx].position.velocity = revolutions * v * joint_signs_[joint_idx];
                    }
                    if (joint_traj_[joint_idx].points[traj_idx].effort.size() == 1) {
                        double t = joint_traj_[joint_idx].points[traj_idx].effort[0];
                        moteus_command_buf_[joint_idx].position.feedforward_torque = t * joint_signs_[joint_idx];
                    }
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
        // std::cout << "rx count " << rx_count << std::endl;

        for (size_t i = 0; i < rx_count; i++)
        {
            int bus = moteus_reply_buf_[i].bus;
            int id = moteus_reply_buf_[i].id;
            size_t joint_idx = joint_uid_to_joint_index_[servo_uid(bus, id)];
            if (moteus_reply_buf_[i].result.mode == moteus::Mode::kFault) {
                RCLCPP_ERROR_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 500 /* [ms] */,
                    "Fault joint: " << joint_names_[joint_idx] << ", fault code: " << moteus_reply_buf_[i].result.fault);
            }
            if (moteus_reply_buf_[i].result.mode == moteus::Mode::kPositionTimeout) {
                RCLCPP_ERROR_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 500 /* [ms] */,
                    "Position timeout joint: " << joint_names_[joint_idx]);
            }
            std::get<0>(moteus_query_res_[joint_idx]) = moteus_reply_buf_[i].result;
            std::get<1>(moteus_query_res_[joint_idx]) = now;
        }
        // publish joint states
        sensor_msgs::msg::JointState msg;
        msg.header.stamp = now;
        std::vector<std::string> joints_not_responding;
        for (size_t joint_idx = 0; joint_idx < nb_joints_; joint_idx++)
        {
            auto joint_name = joint_names_[joint_idx];
            auto res = std::get<0>(moteus_query_res_[joint_idx]);
            auto timestamp = std::get<1>(moteus_query_res_[joint_idx]);
            if (std::isfinite(res.position) && (now - timestamp).seconds() < joint_state_timeout_) {
                msg.name.push_back(joint_name);
                double p = res.position * 2 * M_PI;
                double sign = joint_signs_[joint_idx];
                msg.position.push_back((p - joint_offsets_[joint_idx]) * sign);
                msg.velocity.push_back(res.velocity * 2 * M_PI * sign);
                msg.effort.push_back(res.torque * sign);
                sensor_msgs::msg::Temperature temp_msg;
                temp_msg.temperature = res.temperature;
                temp_pub_[joint_idx]->publish(temp_msg);
                std_msgs::msg::Float32 volt_msg;
                volt_msg.data = res.voltage;
                volt_pub_[joint_idx]->publish(volt_msg);
            } else {
                joints_not_responding.push_back(joint_name);
            }
        }
        joint_pub_->publish(msg);
        if (!joints_not_responding.empty()) {
            // concatenate joints_not_responding
            std::string joint_list_str;
            for (size_t i = 0; i < joints_not_responding.size(); i++) {
                joint_list_str += joints_not_responding[i];
                if (i < joints_not_responding.size() - 1) {
                    joint_list_str += ", ";
                }
            }
            RCLCPP_WARN_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 500 /* [ms] */,
                "Joints not responding: " << joint_list_str);
        }
    }

    void traj_cb(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
    {
        for (size_t msg_joint_idx = 0; msg_joint_idx < msg->joint_names.size(); msg_joint_idx++) {
            auto pos = std::find(joint_names_.begin(), joint_names_.end(), msg->joint_names[msg_joint_idx]);
            if (pos != joint_names_.end()) {
                size_t joint_idx = std::distance(joint_names_.begin(), pos);
                joint_traj_[joint_idx].header = msg->header;
                joint_traj_[joint_idx].points.resize(msg->points.size());
                for (size_t i = 0; i < msg->points.size(); i++) {
                    joint_traj_[joint_idx].points[i].time_from_start = msg->points[i].time_from_start;
                    if (msg_joint_idx < msg->points[i].positions.size()) {
                        joint_traj_[joint_idx].points[i].positions.resize(1);
                        joint_traj_[joint_idx].points[i].positions[0] = msg->points[i].positions[msg_joint_idx];
                    } else {
                        joint_traj_[joint_idx].points[i].positions.resize(0);
                    }
                    if (msg_joint_idx < msg->points[i].velocities.size()) {
                        joint_traj_[joint_idx].points[i].velocities.resize(1);
                        joint_traj_[joint_idx].points[i].velocities[0] = msg->points[i].velocities[msg_joint_idx];
                    } else {
                        joint_traj_[joint_idx].points[i].velocities.resize(0);
                    }
                    if (msg_joint_idx < msg->points[i].effort.size()) {
                        joint_traj_[joint_idx].points[i].effort.resize(1);
                        joint_traj_[joint_idx].points[i].effort[0] = msg->points[i].effort[msg_joint_idx];
                    } else {
                        joint_traj_[joint_idx].points[i].effort.resize(0);
                    }
                }
            }
        }
    }

    static size_t servo_uid(int bus, int id)
    {
        return (bus << 8) + id;
    }

    std::vector<std::string> joint_names_;
    size_t nb_joints_;
    std::vector<double> joint_offsets_;
    std::vector<double> joint_signs_;
    std::map<size_t, size_t> joint_uid_to_joint_index_;
    double joint_state_timeout_;
    double joint_command_timeout_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr> temp_pub_;
    std::vector<rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr> volt_pub_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_sub_;
    std::vector<trajectory_msgs::msg::JointTrajectory> joint_traj_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<moteus::Pi3HatMoteusInterface::ServoCommand> moteus_command_buf_;
    std::vector<moteus::Pi3HatMoteusInterface::ServoReply> moteus_reply_buf_;
    std::shared_ptr<moteus::Pi3HatMoteusInterface> moteus_interface_;
    std::vector<std::tuple<moteus::QueryResult, rclcpp::Time>> moteus_query_res_;
};


RCLCPP_COMPONENTS_REGISTER_NODE(MoteusServo)
