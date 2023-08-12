// IK_CLASS_PIN.h
#ifndef IK_CLASS_PIN_H
#define IK_CLASS_PIN_H

#include <chrono>
#include <math.h>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "pinocchio/algorithm/joint-configuration.hpp"
#pragma GCC diagnostic pop


using namespace std::chrono;

class IKRobot
{
public:
    struct JointState
    {
        std::string name;
        double position;
        double velocity = std::numeric_limits<double>::quiet_NaN();
        double acceleration = std::numeric_limits<double>::quiet_NaN();
        double effort = std::numeric_limits<double>::quiet_NaN();
    };

    struct BodyState
    {
        std::string name;
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d linear_velocity;
        Eigen::Vector3d angular_velocity;
        Eigen::Vector3d linear_acceleration;
        Eigen::Vector3d angular_acceleration;
        typedef enum {FULL_6DOF, POS_ONLY, POS_AXIS} ContraintType;
        ContraintType type;
        Eigen::Vector3d align_axis;  // in the body frame, only used if type == POS_AXIS
        bool in_contact;
        BodyState() {}
        BodyState(const std::string& name, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation)
            : name(name), position(position), orientation(orientation) {
            linear_velocity.setConstant(std::numeric_limits<double>::quiet_NaN());
            angular_velocity.setConstant(std::numeric_limits<double>::quiet_NaN());
            linear_acceleration.setConstant(std::numeric_limits<double>::quiet_NaN());
            angular_acceleration.setConstant(std::numeric_limits<double>::quiet_NaN());
            type = ContraintType::FULL_6DOF;
            align_axis = Eigen::Vector3d::UnitX();
            in_contact = false;
        }
    };

    IKRobot();
    void build_model(const std::string urdf_filename);
    bool has_model() const;
    std::vector<IKRobot::JointState> solve(const std::vector<IKRobot::BodyState>& body_states,
                                                    IKRobot::BodyState odom_baselink,
                                                    std::vector<IKRobot::JointState> &encoder_joint_states,
                                                    std::vector<Eigen::Vector3d> &body_positions_solution,
                                                    std::vector<IKRobot::JointState> &joint_states_for_EL_eq,
                                                    Eigen::VectorXd &gravity_torque,
                                                    Eigen::VectorXd &coriolis_torque,
                                                    Eigen::VectorXd &inertia_torque,
                                                    Eigen::VectorXd &a_foot_computed);

private:

    pinocchio::Model model_;
    Eigen::VectorXd q_;
    Eigen::MatrixXd B_matrix_;
};

#endif
