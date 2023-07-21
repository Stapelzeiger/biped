#ifndef OPTIMIZER_TRAJECTORY_H
#define OPTIMIZER_TRAJECTORY_H
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "OsqpEigen/OsqpEigen.h"

class OptimizerTrajectory
{
public:
    int N_;
    double dt_;
    int nb_total_variables_per_coord_;
    int nb_total_variables_;
    double Ts_;

    OsqpEigen::Solver solver_;
    bool run_optimization_;

public:
    OptimizerTrajectory();

    ~OptimizerTrajectory();
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
    void get_traj_pos_vel(  double dt,
                            double Ts,
                            double T_since_begin_step,
                            Eigen::Vector3d initial_pos,
                            Eigen::Vector3d initial_vel,
                            Eigen::Vector3d final_pos,
                            Eigen::Vector3d final_vel,
                            std::vector<Eigen::Vector3d> &pos_vec,
                            std::vector<Eigen::Vector3d> &vel_vec,
                            std::vector<Eigen::Vector3d> &acc_vec);

};

#endif //OPTIMIZER_FOOT_TRAJECTORY_H
