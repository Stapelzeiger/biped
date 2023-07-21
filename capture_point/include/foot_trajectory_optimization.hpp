#ifndef OPTIMIZER_FOOT_TRAJECTORY_H
#define OPTIMIZER_FOOT_TRAJECTORY_H
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "OsqpEigen/OsqpEigen.h"

class OptimizerFootTrajectory
{
public:
    int N_;
    double Ts_;
    double dt_;
    int nb_total_variables_per_coord_;
    int nb_total_variables_;

    OsqpEigen::Solver solver_;
    bool run_optimization_;

public:
    OptimizerFootTrajectory(double dt, double Ts);
    ~OptimizerFootTrajectory();
    void get_P_and_q_matrices(Eigen::Vector3d opt_weight_pos,
                                 Eigen::Vector3d opt_weight_vel,
                                 Eigen::Vector3d final_pos,
                                 Eigen::Vector3d final_vel,
                                 Eigen::SparseMatrix<double>& P_matrix,
                                 Eigen::VectorXd& q_vec);

    void get_linear_matrix_and_bounds(Eigen::Vector3d initial_pos,
                                      Eigen::Vector3d initial_vel,
                                      double T_since_begin_step,
                                      Eigen::SparseMatrix<double>& A_matrix,
                                      Eigen::VectorXd& l_vec,
                                      Eigen::VectorXd& u_vec);

    void setup_optimization_pb(Eigen::SparseMatrix<double>& P_matrix,
                                Eigen::VectorXd& q_vec,
                                Eigen::SparseMatrix<double>& A_matrix,
                                Eigen::VectorXd& l_vec,
                                Eigen::VectorXd& u_vec);
    Eigen::VectorXd solve_optimization_pb();
    void get_traj_foot_pos_vel(double T_since_begin_step,
                                Eigen::Vector3d initial_pos,
                                Eigen::Vector3d initial_vel,
                                Eigen::Vector3d final_pos,
                                Eigen::Vector3d final_vel,
                                std::vector<Eigen::Vector3d> &foot_pos,
                                std::vector<Eigen::Vector3d> &foot_vel,
                                std::vector<Eigen::Vector3d> &foot_acc);

};

#endif //OPTIMIZER_FOOT_TRAJECTORY_H
