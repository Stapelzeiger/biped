#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "trajectory_optimization.hpp"
#include <fstream>
#include <chrono>

typedef Eigen::Triplet<double> T;

OptimizerTrajectory::OptimizerTrajectory(double dt, double Ts)
{
    Ts_ = Ts;
    nb_total_variables_per_coord_ = 0;
    dt_ = dt;
    solution_opt_start_time_ = 0.0;
    N_ = static_cast<int>(Ts_ / dt_);

    // OSQP Solver Settings:
    solver_.settings()->setWarmStart(false);
    solver_.settings()->setVerbosity(false);
    solver_.settings()->setAbsoluteTolerance(1e-5);
    solver_.settings()->setRelativeTolerance(1e-5);
    solver_.settings()->setMaxIteration(30000);
    // solver_.settings()->getSettings()->time_limit = 0.01;

    run_optimization_ = true;
}

void OptimizerTrajectory::set_position_limits(Eigen::Vector2d pos_x_lims, Eigen::Vector2d pos_y_lims, Eigen::Vector2d pos_z_lims)
{
    pos_x_lims_ = pos_x_lims;
    pos_y_lims_ = pos_y_lims;
    pos_z_lims_ = pos_z_lims;
}

void OptimizerTrajectory::get_P_and_q_matrices(Eigen::Vector3d opt_weight_pos,
                                                    Eigen::Vector3d opt_weight_vel,
                                                    Eigen::Vector3d final_pos,
                                                    Eigen::Vector3d final_vel,
                                                    Eigen::SparseMatrix<double>& P_matrix,
                                                    Eigen::VectorXd& q_vec)
{
    P_matrix.resize(nb_total_variables_, nb_total_variables_);
    P_matrix.setZero();
    q_vec.resize(nb_total_variables_, 1);
    q_vec.setZero();

    for (int i = 0; i < nb_total_variables_per_coord_; i++) {
        P_matrix.insert(i * 3 + 2, i * 3 + 2) = 0.001; // accelerations
    }

    for (int i = 0; i < 3; i++) {
        P_matrix.insert((i+1) * nb_total_variables_per_coord_ - 3, (i+1) * nb_total_variables_per_coord_ - 3) = opt_weight_pos(i); // final pos
        P_matrix.insert((i+1) * nb_total_variables_per_coord_ - 2, (i+1) * nb_total_variables_per_coord_ - 2) = opt_weight_vel(i); // final vel
        q_vec((i+1) * nb_total_variables_per_coord_ - 3) = - opt_weight_pos(i) * final_pos(i);
        q_vec((i+1) * nb_total_variables_per_coord_ - 2) = - opt_weight_vel(i) * final_vel(i);
    }
}

void OptimizerTrajectory::get_linear_matrix_and_bounds(Eigen::Vector3d initial_pos,
                                                              Eigen::Vector3d initial_vel,
                                                              double T_since_begin_step,
                                                              Eigen::SparseMatrix<double>& A_matrix,
                                                              Eigen::VectorXd& l_vec,
                                                              Eigen::VectorXd& u_vec)
{
    // ======== Create A matrix, lower bound and upper bound for initial_conditions ========
    std::vector<T> tripletList;
    int nb_eq_constraints_pos_vel = 6;
    tripletList.push_back(T(0, 0, 1));
    tripletList.push_back(T(1, 1, 1));
    tripletList.push_back(T(2, nb_total_variables_per_coord_, 1));
    tripletList.push_back(T(3, nb_total_variables_per_coord_ + 1, 1));
    tripletList.push_back(T(4, 2 * nb_total_variables_per_coord_, 1));
    tripletList.push_back(T(5, 2 * nb_total_variables_per_coord_ + 1, 1));

    Eigen::MatrixXd l_boundary_pts(nb_eq_constraints_pos_vel, 1);
    Eigen::MatrixXd u_boundary_pts(nb_eq_constraints_pos_vel, 1);
    l_boundary_pts << initial_pos(0), initial_vel(0), initial_pos(1), initial_vel(1), initial_pos(2), initial_vel(2);
    u_boundary_pts << initial_pos(0), initial_vel(0), initial_pos(1), initial_vel(1), initial_pos(2), initial_vel(2);

    // ======== Create A matrix, lower bound and upper bound for dynamics ========
    Eigen::MatrixXd block_dynamics = Eigen::MatrixXd::Zero(2, 6);
    block_dynamics(0, 0) = 1;
    block_dynamics(0, 1) = dt_;
    block_dynamics(0, 3) = -1;
    block_dynamics(1, 1) = 1;
    block_dynamics(1, 2) = dt_;
    block_dynamics(1, 4) = -1;

    int A_dynamics_per_coord_rows = 2 * N_ - 2;
    int A_dynamics_per_coord_cols = 3 * N_;
    int nb_dynamics_constraints = 3 * A_dynamics_per_coord_rows;

    int j;
    for (int bl_idx = 0; bl_idx < 3; bl_idx++) {
        j = 0;
        for (int i = 0; i < N_ - 1; i++) {
            tripletList.push_back(T(nb_eq_constraints_pos_vel + bl_idx * A_dynamics_per_coord_rows + j, bl_idx * A_dynamics_per_coord_cols + i * 3, block_dynamics(0, 0)));
            tripletList.push_back(T(nb_eq_constraints_pos_vel + bl_idx * A_dynamics_per_coord_rows + j, bl_idx * A_dynamics_per_coord_cols + i * 3 + 1, block_dynamics(0, 1)));
            tripletList.push_back(T(nb_eq_constraints_pos_vel + bl_idx * A_dynamics_per_coord_rows + j, bl_idx * A_dynamics_per_coord_cols + i * 3 + 3, block_dynamics(0, 3)));
            tripletList.push_back(T(nb_eq_constraints_pos_vel + bl_idx * A_dynamics_per_coord_rows + j + 1, bl_idx * A_dynamics_per_coord_cols + i * 3 + 1, block_dynamics(1, 1)));
            tripletList.push_back(T(nb_eq_constraints_pos_vel + bl_idx * A_dynamics_per_coord_rows + j + 1, bl_idx * A_dynamics_per_coord_cols + i * 3 + 2, block_dynamics(1, 2)));
            tripletList.push_back(T(nb_eq_constraints_pos_vel + bl_idx * A_dynamics_per_coord_rows + j + 1, bl_idx * A_dynamics_per_coord_cols + i * 3 + 4, block_dynamics(1, 4)));
            j += 2;
        }
    }

    Eigen::MatrixXd l_dynamics = Eigen::MatrixXd::Zero(nb_dynamics_constraints, 1);
    Eigen::MatrixXd u_dynamics = Eigen::MatrixXd::Zero(nb_dynamics_constraints, 1);


    // ======== Create A matrix, lower bound and upper bound for limits ========
    int nb_limits_constraints;
    bool use_pos_limits = true;
    // if (use_limits == true)
    // {
    //     nb_limits_constraints = 2 * nb_total_variables_per_coord_;
    //     for (int i = 0; i < nb_total_variables_per_coord_; i++) {
    //         tripletList.push_back(T(nb_eq_constraints_pos_vel + nb_dynamics_constraints + 2 * i, 3 * i + 1, 1));
    //         tripletList.push_back(T(nb_eq_constraints_pos_vel + nb_dynamics_constraints + 2 * i + 1, 3 * i + 2, 1));
    //     }
    // } else {
    //     nb_limits_constraints = 0;
    // }

    // double a_max = 150;
    // double v_max = 10;
    // Eigen::MatrixXd l_limits = Eigen::MatrixXd::Zero(nb_limits_constraints, 1);
    // Eigen::MatrixXd u_limits = Eigen::MatrixXd::Zero(nb_limits_constraints, 1);
    // for (int i = 0; i < nb_limits_constraints; i += 2) {
    //     l_limits(i) = -v_max;
    //     l_limits(i + 1) = -a_max;
    //     u_limits(i) = v_max;
    //     u_limits(i + 1) = a_max;
    // }

    if (use_pos_limits == true)
    {
        nb_limits_constraints = nb_total_variables_per_coord_;
        for (int i = 0; i < nb_total_variables_per_coord_; i++) {
            tripletList.push_back(T(nb_eq_constraints_pos_vel + nb_dynamics_constraints + i, 3 * i, 1)); // position constraints
        }
    } else {
        nb_limits_constraints = 0;
    }
    Eigen::MatrixXd l_limits = Eigen::MatrixXd::Zero(nb_limits_constraints, 1);
    Eigen::MatrixXd u_limits = Eigen::MatrixXd::Zero(nb_limits_constraints, 1);

    double buffer = 0.1;
    for (int i = 0; i < int(nb_limits_constraints / 3); i++) {
        l_limits(i) = pos_x_lims_[0] - buffer;
        u_limits(i) = pos_x_lims_[1] + buffer;
    }
    for (int i = int(nb_limits_constraints / 3); i < int(2 * nb_limits_constraints / 3); i++) {
        l_limits(i) = pos_y_lims_[0] - buffer;
        u_limits(i) = pos_y_lims_[1] + buffer;
    }
    for (int i = int(2 * nb_limits_constraints / 3); i < int(3 * nb_limits_constraints / 3); i++) {
        l_limits(i) = pos_z_lims_[0];
        u_limits(i) = pos_z_lims_[1];
    }

    // ======== Create A matrix, lower bound and upper bound for z keep ========
    int n_keep = 0;
    int n_start_keep = 0;
    double duration_keep = 33.333/100*Ts_;
    double T_start_keep = 33.333/100*Ts_;

    if (T_since_begin_step <= T_start_keep)
    {
        n_keep = static_cast<int>(duration_keep / dt_); 
        n_start_keep = static_cast<int>((T_start_keep - T_since_begin_step) / dt_);

    }
    if (T_since_begin_step > duration_keep + T_start_keep)
    {
        n_keep = 0;
        n_start_keep = 0;
    }
    if (T_since_begin_step >= T_start_keep && T_since_begin_step <= duration_keep + T_start_keep)
    {
        double time_remaining_to_keep = T_start_keep + duration_keep - T_since_begin_step;
        n_keep = static_cast<int>(time_remaining_to_keep / dt_);
        n_start_keep = 0;
    }
    // todo treat the case in which the foot_z_height is not the same as the initial foot z height when the robot starts walking.

    int nb_keep_constraints = n_keep;
    j = n_start_keep;
    for (int i = 0; i < n_keep; i++)
    {
        tripletList.push_back(T(nb_eq_constraints_pos_vel + nb_dynamics_constraints + nb_limits_constraints + i, 2 * nb_total_variables_per_coord_ + 3 * j, 1));
        j = j + 1;
    }

    double foot_height_keep = desired_foot_raise_height_;
    Eigen::MatrixXd l_keep_foot = foot_height_keep * Eigen::MatrixXd::Ones(n_keep, 1);
    Eigen::MatrixXd u_keep_foot = foot_height_keep * Eigen::MatrixXd::Ones(n_keep, 1);

    A_matrix.resize(nb_eq_constraints_pos_vel + nb_dynamics_constraints + nb_limits_constraints + nb_keep_constraints, nb_total_variables_);
    A_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    A_matrix.makeCompressed();

    l_vec.resize(l_boundary_pts.rows() + l_dynamics.rows() + l_limits.rows() + l_keep_foot.rows(), 1);
    l_vec << l_boundary_pts, l_dynamics,  l_limits, l_keep_foot;

    u_vec.resize(u_boundary_pts.rows() + u_dynamics.rows() + u_limits.rows() + u_keep_foot.rows(), 1);
    u_vec << u_boundary_pts, u_dynamics, u_limits, u_keep_foot;
}

void OptimizerTrajectory::setup_optimization_pb(Eigen::SparseMatrix<double>& P_matrix,
                                                    Eigen::VectorXd& q_vec,
                                                    Eigen::SparseMatrix<double>& A_matrix,
                                                    Eigen::VectorXd& l_vec,
                                                    Eigen::VectorXd& u_vec)
{
    int nb_of_constraints = A_matrix.rows();
    solver_.data()->setNumberOfVariables(nb_total_variables_);
    solver_.data()->setNumberOfConstraints(nb_of_constraints);
    solver_.data()->setHessianMatrix(P_matrix);
    solver_.data()->setGradient(q_vec);
    solver_.data()->setLinearConstraintsMatrix(A_matrix);
    solver_.data()->setLowerBound(l_vec);
    solver_.data()->setUpperBound(u_vec);

    solver_.initSolver();
}

bool OptimizerTrajectory::solve_optimization_pb(Eigen::VectorXd &qp_sol)
{
    Eigen::Vector4d ctr;
    solver_.solveProblem();

    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "[OsqpEigen::Solver::solve] Unable to solve the problem." << std::endl;
        return false;
    }

    // check if the solution is feasible
    if (solver_.getStatus() != OsqpEigen::Status::Solved)
    {
        std::cout << "[OsqpEigen::Solver::solve] The solution is unfeasible." << std::endl;
        return false;
    }

    qp_sol = solver_.getSolution();
    return true;
}

void OptimizerTrajectory::set_desired_foot_raise_height(double desired_foot_raise_height)
{
    desired_foot_raise_height_ = desired_foot_raise_height;
}

void OptimizerTrajectory::set_initial_pos_vel(Eigen::Vector3d initial_pos,
                                                Eigen::Vector3d initial_vel)
{
    initial_pos_ = initial_pos;
    initial_vel_ = initial_vel;
    solution_opt_start_time_ = 0.0;
    solution_opt_pos_.clear();
    solution_opt_vel_.clear();
    solution_opt_acc_.clear();
    traj_opt_computed_ = false;
}

void OptimizerTrajectory::enable_lowering_foot_after_opt_solved(bool enable)
{
    enable_lower_foot_after_opt_solved_ = enable;
}

void OptimizerTrajectory::compute_traj_pos_vel(double T_since_begin_step,
                                            Eigen::Vector3d final_pos,
                                            Eigen::Vector3d &foot_pos,
                                            Eigen::Vector3d &foot_vel,
                                            Eigen::Vector3d &foot_acc)
{
    foot_pos = initial_pos_;
    foot_vel.setZero();
    foot_acc.setZero();

    Eigen::Vector3d opt_weight_pos;
    opt_weight_pos << 5500, 5500, 55000;
    Eigen::Vector3d opt_weight_vel;
    opt_weight_vel << 5500, 5500, 55000;

    Eigen::SparseMatrix<double> P_matrix;
    Eigen::VectorXd q_vec;
    Eigen::SparseMatrix<double> A_matrix;
    Eigen::VectorXd l_vec;
    Eigen::VectorXd u_vec;

    double lower_foot_impact_vel = 0.3;
    Eigen::Vector3d final_vel;
    final_vel << 0.0, 0.0, -lower_foot_impact_vel;

    double T_remaining = Ts_ - T_since_begin_step; // for x and y
    double fraction = 10.0/100.0;

    if (T_remaining > fraction*Ts_ || !traj_opt_computed_)
    {
        N_ = std::round((Ts_ - T_since_begin_step) / dt_);
        nb_total_variables_per_coord_ = 3 * N_; // p, v, a
        nb_total_variables_ = 3 * nb_total_variables_per_coord_; // px, vx, ax, py, vy, ay, pz, vz, az
        if (N_ <= 0)
        {
            std::cout << "Optimization problem not solved because N <= 0" << std::endl;
            return;
        }

        int idx = std::round((T_since_begin_step - solution_opt_start_time_)/dt_);
        if (idx < (int)solution_opt_pos_.size() && idx >= 0)
        {
            initial_pos_ = solution_opt_pos_[idx];
            initial_vel_ = solution_opt_vel_[idx];
        }

        solution_opt_pos_.clear();
        solution_opt_vel_.clear();
        solution_opt_acc_.clear();

        get_P_and_q_matrices(opt_weight_pos, opt_weight_vel, final_pos, final_vel, P_matrix, q_vec);
        get_linear_matrix_and_bounds(initial_pos_, initial_vel_, T_since_begin_step, A_matrix, l_vec, u_vec);
        setup_optimization_pb(P_matrix, q_vec, A_matrix, l_vec, u_vec);
        Eigen::VectorXd sol;
        auto success = solve_optimization_pb(sol);
        if (!success)
        {
            std::cout << "Optimization failed" << std::endl;
            solver_.data()->clearHessianMatrix();
            solver_.data()->clearLinearConstraintsMatrix();
            solver_.clearSolver();
            return;
        }
        for (int i = 0; i < nb_total_variables_per_coord_; i = i + 3)
        {
            Eigen::Vector3d pos;
            Eigen::Vector3d vel;
            Eigen::Vector3d acc;
            pos << sol(i),     sol(i + nb_total_variables_per_coord_),     sol(i + 2 * nb_total_variables_per_coord_);
            vel << sol(i + 1), sol(i + 1 + nb_total_variables_per_coord_), sol(i + 1 + 2 * nb_total_variables_per_coord_);
            acc << sol(i + 2), sol(i + 2 + nb_total_variables_per_coord_), sol(i + 2 + 2 * nb_total_variables_per_coord_);
            solution_opt_pos_.push_back(pos);
            solution_opt_vel_.push_back(vel);
            solution_opt_acc_.push_back(acc);
        }

        foot_pos = solution_opt_pos_[0];
        foot_vel = solution_opt_vel_[0];
        foot_acc = solution_opt_acc_[0];

        solution_opt_start_time_ += dt_ * idx;
        solver_.data()->clearHessianMatrix();
        solver_.data()->clearLinearConstraintsMatrix();
        solver_.clearSolver();
        traj_opt_computed_ = true;

    } else {
        assert(solution_opt_pos_.size() > 0);
        int idx = std::round((T_since_begin_step - solution_opt_start_time_)/dt_);
        if (idx < (int)solution_opt_pos_.size() && idx >= 0)
        {
            foot_pos = solution_opt_pos_[idx];
            foot_vel = solution_opt_vel_[idx];
            foot_acc = solution_opt_acc_[idx];
        } else {
            idx = solution_opt_pos_.size() - 1;
            if (enable_lower_foot_after_opt_solved_ == true)
            {
                foot_pos << solution_opt_pos_[idx](0), solution_opt_pos_[idx](1), solution_opt_pos_[idx](2) - lower_foot_impact_vel * (T_since_begin_step - Ts_);
                foot_vel << 0.0, 0.0, -lower_foot_impact_vel;
            } else {
                foot_pos << solution_opt_pos_[idx](0), solution_opt_pos_[idx](1), solution_opt_pos_[idx](2);
                foot_vel << 0.0, 0.0, 0.0;
            }
            foot_acc << 0.0, 0.0, 0.0;
        }
    }
}

// int main()
// {
//     double dt = 0.005;
//     double Ts = 0.25;
//     double T_since_beginning_of_step = 0.125;
//     double N = int(Ts / dt);
//     OptimizerTrajectory opt(dt, Ts);

//     Eigen::Vector3d final_pos ;
//     Eigen::Vector3d final_vel;
//     final_pos  << 0.2, 0.2, 0.0;
//     final_vel << 0, 0, 0;

//     Eigen::Vector3d initial_pos;
//     Eigen::Vector3d initial_vel;
//     initial_pos << 0.0, 0.0, 0.1;
//     initial_vel << 0.0, 0.0, 0.0;

//     int offset_for_testing = 20;
//     Eigen::Vector3d foot_pos;
//     Eigen::Vector3d foot_vel;
//     Eigen::Vector3d foot_acc;

//     std::vector<Eigen::Vector3d> full_traj_pos;
//     std::vector<Eigen::Vector3d> full_traj_vel;
//     std::vector<Eigen::Vector3d> full_traj_acc;

//     opt.set_initial_pos_vel(initial_pos, initial_vel);
//     for (int i = 0; i < N + offset_for_testing; i++)
//     {
//         std::cout << "iteration nb " << i << " out of " << N << std::endl;
//         opt.compute_traj_pos_vel(T_since_beginning_of_step,
//                                 final_pos,
//                                 foot_pos,
//                                 foot_vel,
//                                 foot_acc);

//         full_traj_pos.push_back(foot_pos);
//         full_traj_vel.push_back(foot_vel);
//         full_traj_acc.push_back(foot_acc);
    
//         std::string file_name;
//         file_name = "/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/capture_point/test/output_traj_cpp_" + std::to_string(i) + ".csv";
//         std::ofstream file(file_name);
//         file << "Position_X,Position_Y,Position_Z,Velocity_X,Velocity_Y,Velocity_Z,Acceleration_X,Acceleration_Y,Acceleration_Z\n";
//         for (unsigned int i = 0; i < opt.solution_opt_pos_.size(); i++)
//         {
//             file << opt.solution_opt_pos_[i](0) << "," << opt.solution_opt_pos_[i](1) << "," << opt.solution_opt_pos_[i](2) << ","
//                 << opt.solution_opt_vel_[i](0) << "," << opt.solution_opt_vel_[i](1) << "," << opt.solution_opt_vel_[i](2) << ","
//                 << opt.solution_opt_acc_[i](0) << "," << opt.solution_opt_acc_[i](1) << "," << opt.solution_opt_acc_[i](2) << "\n";
//         }
//         file.close();
//         T_since_beginning_of_step = T_since_beginning_of_step + dt;
//     }
//     std::string file_name = "/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/capture_point/test/full_traj.csv";
//     std::ofstream file(file_name);
//     file << "Position_X,Position_Y,Position_Z,Velocity_X,Velocity_Y,Velocity_Z,Acceleration_X,Acceleration_Y,Acceleration_Z\n";
//     for (unsigned int i = 0; i < full_traj_pos.size(); i++)
//     {
//         file << full_traj_pos[i](0) << "," << full_traj_pos[i](1) << "," << full_traj_pos[i](2) << ","
//             << full_traj_vel[i](0) << "," << full_traj_vel[i](1) << "," << full_traj_vel[i](2) << ","
//             << full_traj_acc[i](0) << "," << full_traj_acc[i](1) << "," << full_traj_acc[i](2) << "\n";
//     }
//     file.close();
//     return 0;
// }
