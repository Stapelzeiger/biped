// IK_CLASS_PIN.h
#ifndef IK_CLASS_PIN_H
#define IK_CLASS_PIN_H

#include <chrono>
#include <math.h>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
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

        BodyState(const std::string& name, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation)
            : name(name), position(position), orientation(orientation) {
            linear_velocity.setConstant(std::numeric_limits<double>::quiet_NaN());
            angular_velocity.setConstant(std::numeric_limits<double>::quiet_NaN());
            linear_acceleration.setConstant(std::numeric_limits<double>::quiet_NaN());
            angular_acceleration.setConstant(std::numeric_limits<double>::quiet_NaN());
            type = ContraintType::FULL_6DOF;
            align_axis = Eigen::Vector3d::UnitX();
        }
    };

    IKRobot();
    void build_model(const std::string urdf_filename);
    bool has_model() const;
    std::vector<JointState> solve(const std::vector<BodyState>& body_states, std::vector<Eigen::Vector3d> &body_positions_solution);
private:

    pinocchio::Model model_;
    Eigen::VectorXd q_;
};

#endif
