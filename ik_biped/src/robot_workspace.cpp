#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#pragma GCC diagnostic pop
#include <iomanip>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <math.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <time.h>
#include <chrono>
#include <cmath>

using namespace std::placeholders;
using namespace std::chrono_literals;

class Robot
{
public:
    Robot();
    void build_model(const std::string urdf_filename);
    bool has_model() const;
    std::vector<Eigen::Vector3d> compute_robot_workspace();

private:
    pinocchio::Model model_;
    Eigen::VectorXd q_;
};

Robot::Robot()
{
}


void Robot::build_model(const std::string urdf_xml_string)
{
    pinocchio::urdf::buildModelFromXML(urdf_xml_string, pinocchio::JointModelFreeFlyer(), model_);
    q_ = pinocchio::neutral(model_);

    std::cout << "model nq:" << model_.nq << std::endl;
    std::cout << "model nv:" << model_.nv << std::endl;
    std::cout << "model njoints:" << model_.njoints << std::endl;

    std::cout << "model frames:" << std::endl;
    for (auto f : model_.frames) {
        std::cout << "  frame name:" << f.name << std::endl;
        std::cout << "  parent joint:" << model_.names[f.parent] << std::endl;
        std::cout << "  frame placement:" << f.placement << std::endl;
        std::cout << "  frame type" << f.type << std::endl;
        std::cout << "" << std::endl;
    }
    for (auto j : model_.joints) {
        std::cout << "joint idx_q:" << j.idx_q() << std::endl;
        std::cout << "joint idx_v:" << j.idx_v() << std::endl;
        std::cout << "joint nq:" << j.nq() << std::endl;
        std::cout << "joint nv:" << j.nv() << std::endl;
        std::cout << "joint shortname:" << j.shortname() << std::endl;
        // std::cout << "joint type:" << j.type() << std::endl;
        std::cout << "" << std::endl;
    }

}

bool Robot::has_model() const
{
    return model_.njoints > 1;
}

std::vector<Eigen::Vector3d> Robot::compute_robot_workspace()
{
    // initialize q for base_link
    int base_link_joint_id = model_.frames[model_.getFrameId("base_link")].parent;
    const auto &base_link_joint = model_.joints[base_link_joint_id];
    assert(base_link_joint.nq() == 7);
    q_[base_link_joint.idx_q()] = 0.0;
    q_[base_link_joint.idx_q() + 1] = 0.0;
    q_[base_link_joint.idx_q() + 2] = 0.0;
    q_[base_link_joint.idx_q() + 3] = 0.0;
    q_[base_link_joint.idx_q() + 4] = 0.0;
    q_[base_link_joint.idx_q() + 5] = 0.0;
    q_[base_link_joint.idx_q() + 6] = 1.0;

    Eigen::VectorXd q = q_;
    pinocchio::Data data(model_);


    const auto &joint_0 = model_.joints[0];
    double q_0_min = model_.lowerPositionLimit[joint_0.idx_q()];
    double q_0_max = model_.upperPositionLimit[joint_0.idx_q()];

    const auto &joint_1 = model_.joints[1];
    double q_1_min = model_.lowerPositionLimit[joint_1.idx_q()];
    double q_1_max = model_.upperPositionLimit[joint_1.idx_q()];

    const auto &joint_2 = model_.joints[2];
    double q_2_min = model_.lowerPositionLimit[joint_2.idx_q()];
    double q_2_max = model_.upperPositionLimit[joint_2.idx_q()];

    const auto &joint_3 = model_.joints[3];
    double q_3_min = model_.lowerPositionLimit[joint_3.idx_q()];
    double q_3_max = model_.upperPositionLimit[joint_3.idx_q()];

    const auto &joint_4 = model_.joints[4];
    double q_4_min = model_.lowerPositionLimit[joint_4.idx_q()];
    double q_4_max = model_.upperPositionLimit[joint_4.idx_q()];

    std::vector<Eigen::Vector3d> p_left_foot;

    for (double q_0 = q_0_min; q_0 < q_0_max; q_0 += 0.1) {
        for (double q_1 = q_1_min; q_1 < q_1_max; q_1 += 0.1) {
            for (double q_2 = q_2_min; q_2 < q_2_max; q_2 += 0.1) {
                for (double q_3 = q_3_min; q_3 < q_3_max; q_3 += 0.1) {
                    for (double q_4 = q_4_min; q_4 < q_4_max; q_4 += 0.1) {
                        q[joint_0.idx_q()] = q_0;
                        q[joint_1.idx_q()] = q_1;
                        q[joint_2.idx_q()] = q_2;
                        q[joint_3.idx_q()] = q_3;
                        q[joint_4.idx_q()] = q_4;
                        pinocchio::forwardKinematics(model_, data, q);
                        pinocchio::updateFramePlacements(model_, data);
                        auto frame_id = model_.getFrameId("L_ANKLE");
                        const auto &cur_to_world = data.oMf[frame_id];
                        p_left_foot.push_back(cur_to_world.translation());
                        std::cout << "frame placement:" << cur_to_world.translation() << std::endl;
                    }
                }
            }
        }
    }

    return p_left_foot;

}

class RobotWorkspace : public rclcpp::Node
{

public:
    RobotWorkspace() : Node("robot_workspace_node")
    {
        robot_desc_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/robot_description", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(), std::bind(&RobotWorkspace::robot_desc_cb, this, _1));
    
        std::chrono::duration<double> period = 1.0 * 1s;
        timer_ = rclcpp::create_timer(this, this->get_clock(), period, std::bind(&RobotWorkspace::timer_callback, this));

    }

    void timer_callback()
    {
        if (robot_.has_model()){
            file_foot_locations_.open ("data.csv");
            file_foot_locations_ << "x,y,z,\n";
            robot_.compute_robot_workspace();
            file_foot_locations_.close();
        }

        std::ifstream file("data.csv");
        if (file.peek() == std::ifstream::traits_type::eof())
        {
            std::cout << "File is empty" << std::endl;
            rclcpp::shutdown();
        }
    }

    void robot_desc_cb(const std_msgs::msg::String::SharedPtr msg)
    {
        std::cout << "Subscribed to robot description" << std::endl;
        robot_.build_model(msg->data.c_str());
    }

    Robot robot_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_desc_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::ofstream file_foot_locations_;
};




int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotWorkspace>());
    rclcpp::shutdown();
    return 0;
}