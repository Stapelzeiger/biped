#ifndef FOOT_TRAJECTORY_H
#define FOOT_TRAJECTORY_H

#include <iostream>
#include "eigen3/Eigen/Dense"

class SplineTrajectory
{
    // a0 + a1*t + a2*t^2 + a3*t^3
    public:
        SplineTrajectory();
        ~SplineTrajectory();

        Eigen::Vector4d get_spline_coef(double tf, double qi, double qi_dot, double qf, double qf_dot);
        double get_q(Eigen::Vector4d coeff, double t);
        double get_q_dot(Eigen::Vector4d coeff, double t);
        double get_q_ddot(Eigen::Vector4d coeff, double t);
};

class FootTrajectory
{

public:
    Eigen::Vector3d desired_end_position_;
    Eigen::Vector3d initial_position_;
    Eigen::Vector3d computed_foot_pos_;
    Eigen::Vector3d computed_foot_vel_;
    double T_step_;
    double dt_;
    Eigen::Vector<double, 4> coeff_x_spline_;
    Eigen::Vector<double, 4> coeff_y_spline_;
    Eigen::Vector<double, 4> coeff_z_spline_lift_;
    Eigen::Vector<double, 4> coeff_z_spline_lower_;
    bool coeff_lift_computed_ = false;
    double coeff_lift_timestamp_ = 0.0;
    bool coeff_lower_computed_ = false;
    double coeff_lower_timestamp_ = 0.0;
    bool coeff_xy_computed_ = false;
    double coeff_xy_timestamp_ = 0.0;
    int state_in_traj_;
    SplineTrajectory spline_;

public:
    FootTrajectory(double T_step = 0.3, double dt = 0.01);
    ~FootTrajectory();

    void set_desired_end_position(Eigen::Vector3d desired_end_position);
    void set_initial_position(Eigen::Vector3d initial_position);
    void integrate_trajectory_forward(Eigen::Vector3d foot_position, Eigen::Vector3d foot_velocity, Eigen::Vector3d foot_acceleration);
    void get_traj_foot_pos_vel(double T_since_begin_step, Eigen::Vector3d &foot_pos, Eigen::Vector3d &foot_vel, Eigen::Vector3d &foot_acc);
};


#endif