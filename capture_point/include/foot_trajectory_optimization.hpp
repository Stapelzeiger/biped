#ifndef OPTIMIZER_FOOT_TRAJECTORY_H
#define OPTIMIZER_FOOT_TRAJECTORY_H

#include <Eigen/Dense>

class OptimizerFootTrajectory
{
private:
    Eigen::MatrixXd A_matrix_;
    Eigen::MatrixXd P_matrix_;
    Eigen::VectorXd q_vec_;
    Eigen::VectorXd l_vec_;
    Eigen::VectorXd u_vec_;
    int N_;
    double Ts_;
    double dt_;
    double T_since_beginning_of_step_;
    int nb_total_variables_per_coord_;
    int nb_total_variables_;

    Eigen::Vector3d computed_foot_pos_;
    Eigen::Vector3d computed_foot_vel_;

public:
    OptimizerFootTrajectory(double dt, double Ts);
    ~OptimizerFootTrajectory();

    void set_initial_pos_and_vel(Eigen::Vector3d initial_position);

    void update_optimization_matrices();
    void create_optimization_pb();
    Eigen::VectorXd solve();
    void integrate_trajectory_forward(Eigen::Vector3d foot_position, Eigen::Vector3d foot_velocity, Eigen::Vector3d foot_acceleration);

    Eigen::Vector3d getComputedFootPos();
    Eigen::Vector3d getComputedFootVel();
};

#endif //OPTIMIZER_FOOT_TRAJECTORY_H
