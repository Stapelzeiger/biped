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
    pinocchio::Data data(model_);
    pinocchio::forwardKinematics(model_, data, q_);
    pinocchio::updateFramePlacements(model_, data);
    std::cout << "model nq:" << model_.nq << std::endl;
    std::cout << "model nv:" << model_.nv << std::endl;
    std::cout << "model njoints:" << model_.njoints << std::endl;

    std::cout << "model frames:" << std::endl;
    for (auto f : model_.frames) {
        std::cout << "  frame name:" << f.name << std::endl;
        // std::cout << "  parent joint:" << model_.names[f.parent] << std::endl;
        // std::cout << "  frame placement:" << f.placement << std::endl;
        // std::cout << "  frame type" << f.type << std::endl;
        // std::cout << "" << std::endl;
        std::cout << data.oMf[model_.getFrameId(f.name )].translation() << std::endl;

    }
    for (auto j : model_.joints) {
        std::cout << "joint idx_q:" << j.idx_q() << std::endl;
        std::cout << "joint idx_v:" << j.idx_v() << std::endl;
        std::cout << "joint nq:" << j.nq() << std::endl;
        std::cout << "joint nv:" << j.nv() << std::endl;
        std::cout << "joint shortname:" << j.shortname() << std::endl;
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

    std::vector<std::string> end_effector_names;
    std::vector<int> end_effector_joint_ids;
    for (const auto &joint_subtree : model_.subtrees) {
        if (joint_subtree.size() == 1) { // is leaf joint
            end_effector_names.push_back(model_.names[joint_subtree[0]]);
            end_effector_joint_ids.push_back(joint_subtree[0]);
        }
    }

    std::cout << "end_effector_names:" << std::endl;
    for (const auto &end_effector_name : end_effector_names) {
        std::cout << "  " << end_effector_name << std::endl;
    }

    
    std::vector<Eigen::VectorXd> joints_con_to_eff;
    std::vector<Eigen::VectorXd> q_min_list;
    std::vector<Eigen::VectorXd> q_max_list;

    for (const auto &end_effector_name : end_effector_names) {
        auto end_effector_joint_id = model_.getJointId(end_effector_name);
        Eigen::VectorXd connected_joints(5);
        
        connected_joints << end_effector_joint_id - 4, end_effector_joint_id - 3, end_effector_joint_id - 2, end_effector_joint_id - 1, end_effector_joint_id;
        joints_con_to_eff.push_back(connected_joints);

        Eigen::VectorXd q_min_per_foot(5), q_max_per_foot(5);
        int i = 0;
        for (auto j_idx : connected_joints) {
        
            auto &joint = model_.joints[j_idx];
            q_min_per_foot[i] = model_.lowerPositionLimit[joint.idx_q()];
            q_max_per_foot[i] = model_.upperPositionLimit[joint.idx_q()];
            i ++; 

        }
        q_min_list.push_back(q_min_per_foot);
        q_max_list.push_back(q_max_per_foot);

    }

    for (auto joint_list : joints_con_to_eff) {
        std::cout << joint_list.transpose() << std::endl;
    }

    int i = 0;
    for (auto q_min_item : q_min_list) {
        std::cout << "for joint = " << joints_con_to_eff[i].transpose() << " limits are " << q_min_item.transpose()<< std::endl;
        i++;
    }
    i = 0;
    for (auto q_max_item : q_max_list) {
        std::cout << "for joint = " << joints_con_to_eff[i].transpose() << " limits are " << q_max_item.transpose()<< std::endl;
        i++;
    }



    pinocchio::forwardKinematics(model_, data, q);
    pinocchio::updateFramePlacements(model_, data);

    std::vector<Eigen::Vector3d> eff_positions_list;

    // save the locations of the links as well

    std::cout << "model frames:" << std::endl;
    std::cout << model_.frames.size() << std::endl;
    for (auto f : model_.frames) {
        std::cout << "  frame name:" << f.name << std::endl;
        auto positions_frames =  data.oMf[model_.getFrameId(f.name )].translation();
        eff_positions_list.push_back(positions_frames);
    }

    int counter_eff = 0;
    for (const auto &end_effector_name : end_effector_names) {

        for (double q_0 = q_min_list[counter_eff][0]; q_0 < q_max_list[counter_eff][0]; q_0 += 0.1) { // YAW
            for (double q_1 = q_min_list[counter_eff][1]; q_1 < q_max_list[counter_eff][1]; q_1 += 0.1) { // HAA
                for (double q_2 = q_min_list[counter_eff][2]; q_2 < q_max_list[counter_eff][2]; q_2 += 0.1) { // HFE
                    for (double q_3 = q_min_list[counter_eff][3]; q_3 < q_max_list[counter_eff][3]; q_3 += 0.01) { // KFE
                            // std::cout << "q_0 = " << q_0 << "q_1 = " << q_1 << "q_2 = " << q_2 << "q_3 = " << q_3 << "q_4 =" << q_4 << std::endl;
                            std::cout << q_3 << std::endl;
                            auto &joint = model_.joints[end_effector_joint_ids[counter_eff]];
                            q = q_;
                            // std::cout << joint.idx_q() << std::endl;
                            q[joint.idx_q() - 4] = q_0; // YAW
                            q[joint.idx_q() - 3] = q_1; // HAA
                            q[joint.idx_q() - 2] = q_2; // HFE
                            q[joint.idx_q() - 1] = q_3; // KFE
                            q[joint.idx_q()] = 0.0;
                            pinocchio::forwardKinematics(model_, data, q);
                            pinocchio::updateFramePlacements(model_, data);
                            eff_positions_list.push_back(data.oMf[model_.getFrameId(end_effector_name)].translation());
                    }
                }
            }
        }
        
        counter_eff ++;
    }

    return eff_positions_list;

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
        std::vector<Eigen::Vector3d> p_left_foot;
        if (robot_.has_model()){
            file_foot_locations_.open ("data.csv");
            file_foot_locations_ << "x,y,z\n";
            p_left_foot = robot_.compute_robot_workspace();
            for (auto &p : p_left_foot){
                file_foot_locations_ << p[0] << "," << p[1] << "," << p[2] << "\n";
            }
            file_foot_locations_.close();
        }

        std::ifstream file("data.csv");
        if (file.peek() != std::ifstream::traits_type::eof())
        {
            std::cout << "File is not empty!" << std::endl;
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