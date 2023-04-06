#include "foot_trajectory.h"
#include <cmath>
using namespace std;


double get_q(Eigen::Vector<double, 4> coeff, double t)
{
    return coeff(0) + coeff(1)*t + coeff(2)*pow(t,2) + coeff(3)*pow(t,3);
}

double get_q_dot(Eigen::Vector<double, 4> coeff, double t)
{
    return coeff(1) + 2*coeff(2)*t + 3*coeff(3)*pow(t,2);
}

double get_q_ddot(Eigen::Vector<double, 4> coeff, double t)
{
    return 2*coeff(2) + 6*coeff(3)*t;
}


Eigen::Vector<double, 4> get_spline_coef(double tf, double qi, double qi_dot, double qf, double qf_dot)
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

foot_pos_vel_acc_struct get_traj_foot_pos_vel(double T_since_begin_step, double T_step,
                            Eigen::Vector3d current_pos, 
                            Eigen::Vector3d current_vel, 
                            Eigen::Vector3d initial_pos, 
                            Eigen::Vector3d des_pos){

        double lift_foot_hgt = 0.10;
        double lower_foot_impact_vel = 0.4;
        double T_lift = 0.25 * T_step;
        double T_lower = 0.25 * T_step;
        double T_keep = T_step - T_lift - T_lower;
        double delta_h_step = des_pos(2) - initial_pos(2);

        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
        Eigen::Vector3d acc;

        double T_lift_remaining = T_lift - T_since_begin_step;
        double T_keep_remaining = T_step - T_lower - T_since_begin_step;
        double T_lower_remaining = T_step - T_since_begin_step;
        if (T_lift_remaining >= 0) { // here we should have T_step*T_control, because of discretization.
            // lift foot
            Eigen::Vector<double, 4> coeff_z = get_spline_coef(T_lift_remaining, current_pos(2), current_vel(2), initial_pos(2) + lift_foot_hgt, delta_h_step / T_keep);
            pos(2) = get_q(coeff_z, 0.0);
            vel(2) = get_q_dot(coeff_z, 0.0);
            acc(2) = get_q_ddot(coeff_z, 0.0);
        } else if (T_keep_remaining >= 0) {
            // keep foot
            double a = 1 - T_keep_remaining/T_keep;
            pos(2) = initial_pos(2) + lift_foot_hgt + a * delta_h_step;
            std::cout << pos(2) << std::endl;
            vel(2) = delta_h_step / T_keep;
            acc(2) = 0;
        } else if (T_lower_remaining >= 0) {
            // lower foot
            Eigen::Vector<double, 4> coeff_z = get_spline_coef(T_lower_remaining, current_pos(2), current_vel(2), des_pos(2), -lower_foot_impact_vel);
            pos(2) = get_q(coeff_z, 0.0);
            vel(2) = get_q_dot(coeff_z, 0.0);
            acc(2) = get_q_ddot(coeff_z, 0.0);
        } else
        {
            // continue lowering
            pos(2) = current_pos(2); // z = lambda t: current_pos[2] - lower_foot_impact_vel*t
            vel(2) = -lower_foot_impact_vel;
            acc(2) = 0;
        }

        double T_remaining = T_step - T_since_begin_step - T_step*0.1;

        if (T_remaining > 0.005)
        {
            Eigen::Vector<double, 4> coeff_x = get_spline_coef(T_remaining, current_pos(0), current_vel(0), des_pos(0), 0);
            Eigen::Vector<double, 4> coeff_y = get_spline_coef(T_remaining, current_pos(1), current_vel(1), des_pos(1), 0);
            pos(0) = get_q(coeff_x, 0.0);
            pos(1) = get_q(coeff_y, 0.0);
            vel(0) = get_q_dot(coeff_x, 0.0);
            vel(1) = get_q_dot(coeff_y, 0.0);
            acc(0) = get_q_ddot(coeff_x, 0.0);
            acc(1) = get_q_ddot(coeff_y, 0.0);
        }
        else{
            pos(0) = current_pos(0);
            pos(1) = current_pos(1);
            vel(0) = 0.0;
            vel(1) = 0.0;
            acc(0) = 0.0;
            acc(1) = 0.0;
        }

        foot_pos_vel_acc_struct foot_pos_vel_acc;
        foot_pos_vel_acc.pos = pos;
        foot_pos_vel_acc.vel = vel;
        foot_pos_vel_acc.acc = acc;

        return foot_pos_vel_acc;

}