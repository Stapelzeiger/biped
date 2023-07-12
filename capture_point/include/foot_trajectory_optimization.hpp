#ifndef OPTIMIZER_FOOT_TRAJECTORY_H
#define OPTIMIZER_FOOT_TRAJECTORY_H
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "OsqpEigen/OsqpEigen.h"

class OptimizerFootTrajectory
{
public:
    Eigen::SparseMatrix<double> P_matrix_;
    Eigen::VectorXd q_vec_;
    Eigen::SparseMatrix<double> A_matrix_;
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
    OsqpEigen::Solver solver_;
    bool run_optimization_;

public:
    OptimizerFootTrajectory(double dt, double Ts);
    ~OptimizerFootTrajectory();
    bool update_P_and_q_matrices(Eigen::Vector3d opt_weight_pos,
                                 Eigen::Vector3d opt_weight_vel,
                                 Eigen::Vector3d p_N_des,
                                 Eigen::Vector3d v_N_des);

    void update_linear_matrix_and_bounds(Eigen::Vector3d p_0_des,
                                                                  Eigen::Vector3d v_0_des,
                                                                  double v_max,
                                                                  double a_max);

    void update_nb_variables(double T_since_beginning_of_step);

    void create_optimization_pb();
    Eigen::VectorXd solve();
    void integrate_trajectory_forward(Eigen::Vector3d foot_position, Eigen::Vector3d foot_velocity, Eigen::Vector3d foot_acceleration);

    Eigen::Vector3d getComputedFootPos();
    Eigen::Vector3d getComputedFootVel();
};

#endif //OPTIMIZER_FOOT_TRAJECTORY_H
