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
    std::vector<Eigen::Vector3d> solution_opt_pos_;
    std::vector<Eigen::Vector3d> solution_opt_vel_;
    std::vector<Eigen::Vector3d> solution_opt_acc_;
    double solution_opt_start_time_;
    bool enable_lower_foot_after_opt_solved_;

    Eigen::Vector3d initial_pos_;
    Eigen::Vector3d initial_vel_;
    double desired_foot_raise_height_;

    OsqpEigen::Solver solver_;
    bool run_optimization_;
    bool traj_opt_computed_;

    Eigen::Vector2d pos_x_lims_;
    Eigen::Vector2d pos_y_lims_;
    Eigen::Vector2d pos_z_lims_;

public:
    OptimizerTrajectory(double dt, double Ts);
    OptimizerTrajectory(){}
    void set_position_limits(Eigen::Vector2d pos_x_lims, Eigen::Vector2d pos_y_lims, Eigen::Vector2d pos_z_lims);

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

    bool solve_optimization_pb(Eigen::VectorXd &qp_sol);

    bool compute_traj_pos_vel(double T_since_begin_step,
                                Eigen::Vector3d final_pos,
                                Eigen::Vector3d &foot_pos,
                                Eigen::Vector3d &foot_vel,
                                Eigen::Vector3d &foot_acc);

    void set_initial_pos_vel(Eigen::Vector3d initial_pos,
                             Eigen::Vector3d initial_vel);

    void enable_lowering_foot_after_opt_solved(bool enable);

    void set_desired_foot_raise_height(double desired_foot_raise_height);
};

#endif //OPTIMIZER_FOOT_TRAJECTORY_H
