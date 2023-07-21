#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "foot_trajectory_optimization.hpp"
#include <fstream>
#include <chrono>

OptimizerFootTrajectory::OptimizerFootTrajectory(double dt, double Ts)
{
    Ts_ = Ts;
    dt_ = dt;

    N_ = static_cast<int>(Ts_ / dt_);
    nb_total_variables_per_coord_ = 3 * N_; // p, v, a
    nb_total_variables_ = 3 * nb_total_variables_per_coord_; // px, vx, ax, py, vy, ay, pz, vz, az

    // OSQP Solver Settings:
    solver_.settings()->setWarmStart(false);
    solver_.settings()->setVerbosity(true);
    solver_.settings()->setAbsoluteTolerance(1e-5);
    solver_.settings()->setRelativeTolerance(1e-5);
    solver_.settings()->setMaxIteration(30000);
    // solver_.settings()->getSettings()->time_limit = 0.01;

    run_optimization_ = true;
}

OptimizerFootTrajectory::~OptimizerFootTrajectory()
{
}

void OptimizerFootTrajectory::get_P_and_q_matrices(Eigen::Vector3d opt_weight_pos,
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

void OptimizerFootTrajectory::get_linear_matrix_and_bounds(Eigen::Vector3d initial_pos,
                                                              Eigen::Vector3d initial_vel,
                                                              double T_since_begin_step,
                                                              Eigen::SparseMatrix<double>& A_matrix,
                                                              Eigen::VectorXd& l_vec,
                                                              Eigen::VectorXd& u_vec)
{
    // ======== Create A matrix ========
    Eigen::MatrixXd A_eq_pos_vel_des = Eigen::MatrixXd::Zero(6, nb_total_variables_);
    A_eq_pos_vel_des(0, 0) = 1;
    A_eq_pos_vel_des(1, 1) = 1;
    A_eq_pos_vel_des(2, nb_total_variables_per_coord_) = 1;
    A_eq_pos_vel_des(3, nb_total_variables_per_coord_ + 1) = 1;
    A_eq_pos_vel_des(4, 2 * nb_total_variables_per_coord_) = 1;
    A_eq_pos_vel_des(5, 2 * nb_total_variables_per_coord_ + 1) = 1;

    Eigen::MatrixXd block_dynamics = Eigen::MatrixXd::Zero(2, 6);
    block_dynamics(0, 0) = 1;
    block_dynamics(0, 1) = dt_;
    block_dynamics(0, 3) = -1;
    block_dynamics(1, 1) = 1;
    block_dynamics(1, 2) = dt_;
    block_dynamics(1, 4) = -1;

    int A_dynamics_per_coordinate_rows = 2 * N_ - 2;
    int A_dynamics_per_coordinate_cols = 3 * N_;
    Eigen::MatrixXd A_dynamics_per_coordinate = Eigen::MatrixXd::Zero(A_dynamics_per_coordinate_rows, A_dynamics_per_coordinate_cols);

    int j = 0;
    for (int i = 0; i < N_ - 1; i++) {
        A_dynamics_per_coordinate.block<2, 6>(j, i * 3) = block_dynamics;
        j += 2;
    }

    Eigen::MatrixXd A_dynamics(3 * A_dynamics_per_coordinate_rows, nb_total_variables_);
    A_dynamics.setZero();

    for (int i = 0; i < 3; i++) {
        A_dynamics.block(i * A_dynamics_per_coordinate_rows, i * A_dynamics_per_coordinate_cols,
                        A_dynamics_per_coordinate_rows, A_dynamics_per_coordinate_cols) = A_dynamics_per_coordinate;
    }

    Eigen::MatrixXd A_limits = Eigen::MatrixXd::Zero(nb_total_variables_per_coord_ * 1, nb_total_variables_);
    for (int i = 0; i < nb_total_variables_per_coord_; i++) {
        A_limits(i, 3 * i + 2) = 1.0;
    }

    int n_keep;
    int n_start_keep;
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
        n_keep = static_cast<int>(time_remaining_to_keep / dt_); // not correct
        n_start_keep = 0;
    }

    std::cout << "n keep" << n_keep << std::endl;
    std::cout << "T_since_begin_step = " << T_since_begin_step << std::endl;
    Eigen::MatrixXd A_keep_foot = Eigen::MatrixXd::Zero(n_keep, nb_total_variables_);

    j = n_start_keep;
    for (int i = 0; i < n_keep; i++)
    {
        A_keep_foot(i, 2*nb_total_variables_per_coord_ + 3 * j) = 1;
        j = j + 1;
    }

    Eigen::MatrixXd A_matrix_dense;
    A_matrix_dense.resize(A_eq_pos_vel_des.rows() + A_dynamics.rows() + A_keep_foot.rows(), nb_total_variables_);
    A_matrix_dense.setZero();
    A_matrix_dense.topRows(A_eq_pos_vel_des.rows()) = A_eq_pos_vel_des;
    A_matrix_dense.middleRows(A_eq_pos_vel_des.rows(), A_dynamics.rows()) = A_dynamics;
    A_matrix_dense.bottomRows(A_keep_foot.rows()) = A_keep_foot;

    A_matrix = A_matrix_dense.sparseView();

    // ======== Create l, u matrices ========
    Eigen::MatrixXd l_boundary_pts(6, 1);
    Eigen::MatrixXd u_boundary_pts(6, 1);
    
    l_boundary_pts << initial_pos(0), initial_vel(0), initial_pos(1), initial_vel(1), initial_pos(2), initial_vel(2);
    u_boundary_pts << initial_pos(0), initial_vel(0), initial_pos(1), initial_vel(1), initial_pos(2), initial_vel(2);

    Eigen::MatrixXd l_dynamics = Eigen::MatrixXd::Zero(A_dynamics.rows(), 1);
    Eigen::MatrixXd u_dynamics = Eigen::MatrixXd::Zero(A_dynamics.rows(), 1);

    Eigen::MatrixXd l_limits = Eigen::MatrixXd::Zero(3 * N_, 1);
    Eigen::MatrixXd u_limits = Eigen::MatrixXd::Zero(3 * N_, 1);
    double a_max = 1000;
    for (int i = 0; i < 3 * N_; i = i + 1) {

        l_limits(i) = -a_max;
        u_limits(i) = a_max;
    }
    double foot_height_keep = 0.2;
    Eigen::MatrixXd l_keep_foot = foot_height_keep * Eigen::MatrixXd::Ones(n_keep, 1);
    Eigen::MatrixXd u_keep_foot = foot_height_keep * Eigen::MatrixXd::Ones(n_keep, 1);

    l_vec.resize(l_boundary_pts.rows() + l_dynamics.rows() + l_keep_foot.rows(), 1);
    l_vec << l_boundary_pts, l_dynamics, l_keep_foot;

    u_vec.resize(u_boundary_pts.rows() + u_dynamics.rows() + u_keep_foot.rows(), 1);
    u_vec << u_boundary_pts, u_dynamics, u_keep_foot;
}

void OptimizerFootTrajectory::setup_optimization_pb(Eigen::SparseMatrix<double>& P_matrix,
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

Eigen::VectorXd OptimizerFootTrajectory::solve_optimization_pb()
{
    Eigen::Vector4d ctr;
    Eigen::VectorXd qp_sol;
    solver_.solveProblem();

    qp_sol = solver_.getSolution();
    return qp_sol;
}

void OptimizerFootTrajectory::get_traj_foot_pos_vel(double T_since_begin_step,
                                                    Eigen::Vector3d initial_pos,
                                                    Eigen::Vector3d initial_vel,
                                                    Eigen::Vector3d final_pos,
                                                    Eigen::Vector3d final_vel,
                                                    std::vector<Eigen::Vector3d> &foot_pos,
                                                    std::vector<Eigen::Vector3d> &foot_vel,
                                                    std::vector<Eigen::Vector3d> &foot_acc)
{
    Eigen::Vector3d opt_weight_pos;
    opt_weight_pos << 5500, 5500, 55000;
    Eigen::Vector3d opt_weight_vel;
    opt_weight_vel << 5500, 5500, 55000;

    Eigen::SparseMatrix<double> P_matrix;
    Eigen::VectorXd q_vec;
    Eigen::SparseMatrix<double> A_matrix;
    Eigen::VectorXd l_vec;
    Eigen::VectorXd u_vec;

    N_ = static_cast<int>((Ts_ - T_since_begin_step) / dt_);
    nb_total_variables_per_coord_ = 3 * N_; // p, v, a
    nb_total_variables_ = 3 * nb_total_variables_per_coord_; // px, vx, ax, py, vy, ay, pz, vz, az

    get_P_and_q_matrices(opt_weight_pos, opt_weight_vel, final_pos, final_vel, P_matrix, q_vec);
    get_linear_matrix_and_bounds(initial_pos, initial_vel, T_since_begin_step, A_matrix, l_vec, u_vec);
    setup_optimization_pb(P_matrix, q_vec, A_matrix, l_vec, u_vec);
    Eigen::VectorXd sol;
    sol = solve_optimization_pb();
    for (int i = 0; i < nb_total_variables_per_coord_; i = i + 3)
    {
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
        Eigen::Vector3d acc;
        pos << sol(i),     sol(i + nb_total_variables_per_coord_),     sol(i + 2 * nb_total_variables_per_coord_);
        vel << sol(i + 1), sol(i + 1 + nb_total_variables_per_coord_), sol(i + 1 + 2 * nb_total_variables_per_coord_);
        acc << sol(i + 2), sol(i + 2 + nb_total_variables_per_coord_), sol(i + 2 + 2 * nb_total_variables_per_coord_);
        foot_pos.push_back(pos);
        foot_vel.push_back(vel);
        foot_acc.push_back(acc);
    }
     solver_.data()->clearHessianMatrix();
     solver_.data()->clearLinearConstraintsMatrix();
     solver_.clearSolver();
}


int main()
{
    double dt = 0.01;
    double Ts = 0.3;
    double T_since_beginning_of_step = 0.0;
    double N = int(Ts / dt);
    OptimizerFootTrajectory opt(dt, Ts);

    Eigen::Vector3d final_pos ;
    Eigen::Vector3d final_vel;
    final_pos  << 0.2, 0.2, 0.0;
    final_vel << 0, 0, 0;

    Eigen::Vector3d initial_pos;
    Eigen::Vector3d initial_vel;
    initial_pos << 0.0, 0.0, 0.0;
    initial_vel << 0.0, 0.0, 0.0;

    int offset_for_testing = 0;

    for (int i = 0; i < N + offset_for_testing; i++)
    {
        std::vector<Eigen::Vector3d> foot_position;
        std::vector<Eigen::Vector3d> foot_velocity;
        std::vector<Eigen::Vector3d> foot_acceleration;
        if (T_since_beginning_of_step > 90.0/100.0*Ts)
        {
            std::cout << "stop opt pb!" << std::endl;
            // integrate forward open-loop
            Eigen::Vector3d pos;
            Eigen::Vector3d vel;
            Eigen::Vector3d acc;
            pos << initial_pos(0), initial_pos(1), initial_pos(2);
            vel << initial_vel(0), initial_vel(1), initial_vel(2);
            acc << 0, 0, 0;

            for (int j = 0; j < N + offset_for_testing - i; j++)
            {
                pos = pos + dt * vel;
                vel = vel + dt * acc;
                foot_position.push_back(pos);
                foot_velocity.push_back(vel);
                foot_acceleration.push_back(acc);
            }

        }
        else
        {
            opt.get_traj_foot_pos_vel(T_since_beginning_of_step,
                                    initial_pos,
                                    initial_vel,
                                    final_pos,
                                    final_vel,
                                    foot_position,
                                    foot_velocity,
                                    foot_acceleration);
        }

        std::string file_name;
        file_name = "/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/capture_point/test/output_foot_traj_cpp_" + std::to_string(i) + ".csv";
        std::ofstream file(file_name);
        file << "Position_X,Position_Y,Position_Z,Velocity_X,Velocity_Y,Velocity_Z,Acceleration_X,Acceleration_Y,Acceleration_Z\n";
        for (unsigned int i = 0; i < foot_position.size(); i++)
        {
            file << foot_position[i](0) << "," << foot_position[i](1) << "," << foot_position[i](2) << ","
                << foot_velocity[i](0) << "," << foot_velocity[i](1) << "," << foot_velocity[i](2) << ","
                << foot_acceleration[i](0) << "," << foot_acceleration[i](1) << "," << foot_acceleration[i](2) << "\n";
        }
        file.close();

        initial_pos = foot_position[1];
        initial_vel = foot_velocity[1];

        T_since_beginning_of_step = T_since_beginning_of_step + dt;

    }



    return 0;
}
