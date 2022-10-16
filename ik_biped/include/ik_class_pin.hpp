// IK_CLASS_PIN.h
#ifndef IK_CLASS_PIN_H
#define IK_CLASS_PIN_H

#include <chrono>
#include <math.h>
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"

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

private:
    pinocchio::Model model;
};

#endif