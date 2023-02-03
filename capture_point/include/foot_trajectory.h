#ifndef FOOT_TRAJECTORY_H
#define FOOT_TRAJECTORY_H

#include <iostream>
#include "eigen3/Eigen/Dense"

struct foot_pos_vel_acc_struct { 
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Eigen::Vector3d acc;
}; 

double get_q(Eigen::Vector<double, 4> coeff, double t);
double get_q_dot(Eigen::Vector<double, 4> coeff, double t);
double get_q_ddot(Eigen::Vector<double, 4> coeff, double t);
Eigen::Vector<double, 4> get_spline_coef(double T, double q0, double q_dot0, 
                                                   double qf, double q_dotf);


foot_pos_vel_acc_struct get_traj_foot_pos_vel(double T_since_begin_step, 
                            double T_step,
                            Eigen::Vector3d current_pos, 
                            Eigen::Vector3d current_vel, 
                            Eigen::Vector3d initial_pos, 
                            Eigen::Vector3d des_pos);

#endif