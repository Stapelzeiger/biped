#include "foot_trajectory.h"
#include <cmath>
using namespace std;

FootTrajectory::FootTrajectory()
{
}

FootTrajectory::FootTrajectory(double T_step, double dt)
{
    T_step_ = T_step;
    dt_ = dt;
}

FootTrajectory::~FootTrajectory()
{
}

void FootTrajectory::set_desired_end_position(Eigen::Vector3d desired_end_position)
{
    desired_end_position_ = desired_end_position;
}

void FootTrajectory::set_initial_position(Eigen::Vector3d initial_position)
{
    initial_position_ = initial_position;
    coeff_lift_computed_ = false;
    coeff_lift_timestamp_ = 0.0;
    coeff_lower_computed_ = false;
    coeff_lower_timestamp_ = 0.0;
    coeff_xy_computed_ = false;
    coeff_xy_timestamp_ = 0.0;
    computed_foot_pos_ = initial_position;
    computed_foot_vel_ = Eigen::Vector3d::Zero();
}

double FootTrajectory::get_q(Eigen::Vector<double, 4> coeff, double t)
{
    return coeff(0) + coeff(1)*t + coeff(2)*pow(t,2) + coeff(3)*pow(t,3);
}

double FootTrajectory::get_q_dot(Eigen::Vector<double, 4> coeff, double t)
{
    return coeff(1) + 2*coeff(2)*t + 3*coeff(3)*pow(t,2);
}

double FootTrajectory::get_q_ddot(Eigen::Vector<double, 4> coeff, double t)
{
    return 2*coeff(2) + 6*coeff(3)*t;
}

void FootTrajectory::integrate_trajectory_forward(Eigen::Vector3d foot_position, Eigen::Vector3d foot_velocity, Eigen::Vector3d foot_acceleration)
{
    for (int i = 0; i < 3; i++)
    {
        computed_foot_pos_(i) = foot_velocity(i) * dt_ + foot_position(i);
        computed_foot_vel_(i) = foot_acceleration(i) * dt_ + foot_velocity(i);
    }
}

Eigen::Vector<double, 4> FootTrajectory::get_spline_coef(double tf, double qi, double qi_dot, double qf, double qf_dot)
{
    Eigen::Vector<double, 4> coeff; // a0 a1 a2 a3
    coeff(0) = qi;
    coeff(1) = qi_dot;

    Eigen::MatrixXd mat(2,2);
    mat(0,0) = pow(tf, 3); mat(0,1) = pow(tf, 2);
    mat(1,0) = 3*pow(tf,2); mat(1,1) = 2*tf;
    Eigen::MatrixXd mat_inv = mat.inverse();

    Eigen::VectorXd vec(2);
    vec(0) = qf - qi - qi_dot*tf;
    vec(1) = qf_dot - qi_dot;

    Eigen::VectorXd vec2 = mat_inv * vec;
    coeff(3) = vec2(0);
    coeff(2) = vec2(1);

    return coeff;
}

void FootTrajectory::get_traj_foot_pos_vel(double T_since_begin_step, Eigen::Vector3d &foot_pos, Eigen::Vector3d &foot_vel, Eigen::Vector3d &foot_acc)
{
    double lift_foot_hgt = 0.10;
    double lower_foot_impact_vel = 0.4;
    double T_lift = 0.40 * T_step_;
    double T_lower = 0.40 * T_step_;
    double T_keep = T_step_ - T_lift - T_lower;
    double delta_h_step = desired_end_position_(2) - initial_position_(2);

    double T_lift_remaining = T_lift - T_since_begin_step;
    double T_keep_remaining = T_step_ - T_lower - T_since_begin_step;
    double T_lower_remaining = T_step_ - T_since_begin_step;
    double T_remaining = T_step_ - T_since_begin_step - T_step_*0.1; // for x and y
    double fraction = 10.0/100.0;

    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Eigen::Vector3d acc;

    if (T_lift_remaining >= 0) {
        // lift foot
        if (T_lift_remaining > fraction*T_lift || !coeff_lift_computed_){
            coeff_z_spline_lift_ = get_spline_coef(T_lift_remaining, computed_foot_pos_(2), computed_foot_vel_(2), initial_position_(2) + lift_foot_hgt, delta_h_step / T_keep);
            coeff_lift_computed_ = true;
            coeff_lift_timestamp_ = T_lift_remaining;
        }
        pos(2) = get_q(coeff_z_spline_lift_, coeff_lift_timestamp_ - T_lift_remaining);
        vel(2) = get_q_dot(coeff_z_spline_lift_, coeff_lift_timestamp_ - T_lift_remaining);
        acc(2) = get_q_ddot(coeff_z_spline_lift_, coeff_lift_timestamp_ - T_lift_remaining);
        state_in_traj_ = 0;
    } else if (T_keep_remaining >= 0) {
        // keep foot
        double a = 1 - T_keep_remaining/T_keep;
        pos(2) = initial_position_(2) + lift_foot_hgt + a * delta_h_step;
        vel(2) = delta_h_step / T_keep;
        acc(2) = 0;
        state_in_traj_ = 1;

    } else if (T_lower_remaining >= 0) {
        state_in_traj_ = 2;
        // lower foot
        if (T_lower_remaining > fraction*T_lower || !coeff_lower_computed_){
            coeff_z_spline_lower_ = get_spline_coef(T_lower_remaining, computed_foot_pos_(2), computed_foot_vel_(2), desired_end_position_(2), -lower_foot_impact_vel);
            coeff_lower_computed_ = true;
            coeff_lower_timestamp_ = T_lower_remaining;
        }
        pos(2) = get_q(coeff_z_spline_lower_, coeff_lower_timestamp_ - T_lower_remaining);
        vel(2) = get_q_dot(coeff_z_spline_lower_, coeff_lower_timestamp_ - T_lower_remaining);
        acc(2) = get_q_ddot(coeff_z_spline_lower_, coeff_lower_timestamp_ - T_lower_remaining);
    } else
    {
        // continue lowering
        pos(2) = computed_foot_pos_(2); // z = lambda t: computed_foot_pos_[2] - lower_foot_impact_vel*t
        vel(2) = -lower_foot_impact_vel;
        acc(2) = 0;
    }

    if (T_remaining > fraction*T_step_ || !coeff_xy_computed_)
    {
        coeff_x_spline_ = get_spline_coef(T_remaining, computed_foot_pos_(0), computed_foot_vel_(0), desired_end_position_(0), 0);
        coeff_y_spline_ = get_spline_coef(T_remaining, computed_foot_pos_(1), computed_foot_vel_(1), desired_end_position_(1), 0);
        pos(0) = get_q(coeff_x_spline_, 0.0);
        pos(1) = get_q(coeff_y_spline_, 0.0);
        vel(0) = get_q_dot(coeff_x_spline_, 0.0);
        vel(1) = get_q_dot(coeff_y_spline_, 0.0);
        acc(0) = get_q_ddot(coeff_x_spline_, 0.0);
        acc(1) = get_q_ddot(coeff_y_spline_, 0.0);
        coeff_xy_computed_ = true;
        coeff_xy_timestamp_ = T_remaining;
    }else{
        pos(0) = get_q(coeff_x_spline_, coeff_xy_timestamp_ - T_remaining);
        pos(1) = get_q(coeff_y_spline_, coeff_xy_timestamp_ - T_remaining);
        vel(0) = get_q_dot(coeff_x_spline_, coeff_xy_timestamp_ - T_remaining);
        vel(1) = get_q_dot(coeff_y_spline_, coeff_xy_timestamp_ - T_remaining);
        acc(0) = get_q_ddot(coeff_x_spline_, coeff_xy_timestamp_ - T_remaining);
        acc(1) = get_q_ddot(coeff_y_spline_, coeff_xy_timestamp_ - T_remaining);
    }

    integrate_trajectory_forward(pos, vel, acc);

    foot_pos = pos;
    foot_vel = vel;
    foot_acc = acc;
}