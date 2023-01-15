#ifndef FOOT_TRAJECTORY_H
#define FOOT_TRAJECTORY_H

#include <iostream>
#include "eigen3/Eigen/Dense"

struct foot_pos_vel_acc_struct { 
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Eigen::Vector3d acc;
}; 


foot_pos_vel_acc_struct get_swing_foot_pos_vel(double T_since_begin_step, 
                            double T_step,
                            Eigen::Vector3d current_pos, 
                            Eigen::Vector3d current_vel, 
                            Eigen::Vector3d initial_pos, 
                            Eigen::Vector3d des_pos);

#endif