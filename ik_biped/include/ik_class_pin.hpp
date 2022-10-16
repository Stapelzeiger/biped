// IK_CLASS_PIN.h
#ifndef IK_CLASS_PIN_H
#define IK_CLASS_PIN_H

#include <chrono>
#include <math.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#pragma GCC diagnostic pop

using namespace std::chrono;
const double eps = 1e-3;
const int IT_MAX = 500;
const double DT = 0.1;
const double damp = 1e-5;

class IKRobot
{
public:
    IKRobot();
    void build_model(const std::string urdf_filename);
    int get_size_q();
    int get_size_q_dot();
    Eigen::VectorXd get_desired_q(Eigen::VectorXd q, Eigen::Vector3d pos_foot_des, double yaw_angle, std::string joint_name);

private:
    pinocchio::Model model;
};

#endif
