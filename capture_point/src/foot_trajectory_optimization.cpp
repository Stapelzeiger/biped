#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "foot_trajectory_optimization.hpp"
#include <fstream>

OptimizerFootTrajectory::OptimizerFootTrajectory(double dt, double Ts)
{
    Ts_ = Ts;
    dt_ = dt;
    T_since_beginning_of_step_ = 0.0; // todo treat the case when T since beginning step = 0.5

    N_ = static_cast<int>(Ts_ / dt_);
    nb_total_variables_per_coord_ = 3 * N_; // p, v, a
    nb_total_variables_ = 3 * nb_total_variables_per_coord_; // x, y, z

    // OSQP Solver Settings:
    solver_.settings()->setWarmStart(false);
    solver_.settings()->setVerbosity(true);
    solver_.settings()->setAbsoluteTolerance(1e-4);
    solver_.settings()->setRelativeTolerance(1e-4);
    solver_.settings()->setMaxIteration(6000);

    run_optimization_ = true;
}

OptimizerFootTrajectory::~OptimizerFootTrajectory()
{
}

void OptimizerFootTrajectory::update_nb_variables(double T_since_beginning_of_step)
{
    N_ = N_ - 1;
    T_since_beginning_of_step_ = T_since_beginning_of_step;
    nb_total_variables_per_coord_ = 3 * N_; // p, v, a
    nb_total_variables_ = 3 * nb_total_variables_per_coord_; // x, y, z

    if (T_since_beginning_of_step_ > 90/100*Ts_) {
        std::cout << "Freeze the optimization problem and keep integrating open loop" << std::endl;
        run_optimization_ = false;
    }

    if (T_since_beginning_of_step_ == 0.0) {
        std::cout << "Reset the optimization problem" << std::endl;
        N_ = static_cast<int>(Ts_ / dt_);
        run_optimization_ = true;
    }

}

void OptimizerFootTrajectory::update_P_and_q_matrices(Eigen::Vector3d opt_weight_pos,
                                                    Eigen::Vector3d opt_weight_vel,
                                                    Eigen::Vector3d p_N_des,
                                                    Eigen::Vector3d v_N_des)
{
    P_matrix_.resize(nb_total_variables_, nb_total_variables_);
    P_matrix_.setZero();
    q_vec_.resize(nb_total_variables_, 1);
    q_vec_.setZero();

    for (int i = 0; i < nb_total_variables_per_coord_; i++) {
        P_matrix_.insert(i * 3 + 2, i * 3 + 2) = 0.001; // accelerations
    }

    for (int i = 0; i < 3; i++) {
        P_matrix_.insert((i+1) * nb_total_variables_per_coord_ - 3, (i+1) * nb_total_variables_per_coord_ - 3) = opt_weight_pos(i); // final pos
        P_matrix_.insert((i+1) * nb_total_variables_per_coord_ - 2, (i+1) * nb_total_variables_per_coord_ - 2) = opt_weight_vel(i); // final vel
        q_vec_((i+1) * nb_total_variables_per_coord_ - 3) = - opt_weight_pos(i) * p_N_des(i);
        q_vec_((i+1) * nb_total_variables_per_coord_ - 2) = - opt_weight_vel(i) * v_N_des(i);
    }

}

void OptimizerFootTrajectory::update_linear_matrix_and_bounds(Eigen::Vector3d p_0_des,
                                                              Eigen::Vector3d v_0_des,
                                                              double v_max,
                                                              double a_max)
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

    Eigen::MatrixXd block = Eigen::MatrixXd::Zero(2, 3);
    block(0, 1) = 1;
    block(1, 2) = 1;
    Eigen::MatrixXd A_limits = Eigen::MatrixXd::Zero(nb_total_variables_per_coord_ * 2, nb_total_variables_);

    for (int i = 0; i < nb_total_variables_per_coord_; i++) {
        A_limits.block<2, 3>(2 * i, 3 * i) = block;
    }

    double T_keep = 33.333/100*Ts_;
    double T_start_keep = 33.333/100*Ts_;
    int n_keep = static_cast<int>(T_keep / dt_);
    int n_start_keep = static_cast<int>(T_start_keep / dt_);
    Eigen::MatrixXd A_keep_foot = Eigen::MatrixXd::Zero(n_keep, nb_total_variables_);

    j = n_start_keep;
    for (int i = 0; i < n_keep; i++)
    {
        A_keep_foot(i, 2*nb_total_variables_per_coord_ + 3 * j) = 1;
        j = j + 1;
    }

    Eigen::MatrixXd A_matrix_dense;
    A_matrix_dense.resize(A_eq_pos_vel_des.rows() + A_dynamics.rows() + A_limits.rows() + A_keep_foot.rows(), nb_total_variables_);
    A_matrix_dense.setZero();
    A_matrix_dense.topRows(A_eq_pos_vel_des.rows()) = A_eq_pos_vel_des;
    A_matrix_dense.middleRows(A_eq_pos_vel_des.rows(), A_dynamics.rows()) = A_dynamics;
    A_matrix_dense.middleRows(A_eq_pos_vel_des.rows() + A_dynamics.rows(), A_limits.rows()) = A_limits;
    A_matrix_dense.bottomRows(A_keep_foot.rows()) = A_keep_foot;

    A_matrix_ = A_matrix_dense.sparseView();

    // ======== Create l, u matrices ========
    Eigen::MatrixXd l_boundary_pts(6, 1);
    Eigen::MatrixXd u_boundary_pts(6, 1);
    
    l_boundary_pts << p_0_des(0), v_0_des(0), p_0_des(1), v_0_des(1), p_0_des(2), v_0_des(2);
    u_boundary_pts << p_0_des(0), v_0_des(0), p_0_des(1), v_0_des(1), p_0_des(2), v_0_des(2);

    Eigen::MatrixXd l_dynamics = Eigen::MatrixXd::Zero(A_dynamics.rows(), 1);
    Eigen::MatrixXd u_dynamics = Eigen::MatrixXd::Zero(A_dynamics.rows(), 1);

    Eigen::MatrixXd l_limits = Eigen::MatrixXd::Zero(3 * 2 * N_, 1);
    Eigen::MatrixXd u_limits = Eigen::MatrixXd::Zero(3 * 2 * N_, 1);
    for (int i = 0; i < 3 * 2 * N_; i += 2) {
        l_limits(i) = -v_max;
        l_limits(i + 1) = -a_max;
        u_limits(i) = v_max;
        u_limits(i + 1) = a_max;
    }

    double foot_height_keep = 0.2;
    Eigen::MatrixXd l_keep_foot = foot_height_keep * Eigen::MatrixXd::Ones(n_keep, 1);
    Eigen::MatrixXd u_keep_foot = foot_height_keep * Eigen::MatrixXd::Ones(n_keep, 1);


    l_vec_.resize(l_boundary_pts.rows() + l_dynamics.rows() + l_limits.rows() + l_keep_foot.rows(), 1);
    l_vec_ << l_boundary_pts, l_dynamics, l_limits, l_keep_foot;

    u_vec_.resize(u_boundary_pts.rows() + u_dynamics.rows() + u_limits.rows() + u_keep_foot.rows(), 1);
    u_vec_ << u_boundary_pts, u_dynamics, u_limits, u_keep_foot;
}

void OptimizerFootTrajectory::create_optimization_pb()
{
    solver_.data()->setNumberOfVariables(nb_total_variables_);
    int nb_of_constraints = A_matrix_.rows();
    solver_.data()->setNumberOfConstraints(nb_of_constraints);
    solver_.data()->setHessianMatrix(P_matrix_);
    solver_.data()->setGradient(q_vec_);
    solver_.data()->setLinearConstraintsMatrix(A_matrix_);
    solver_.data()->setLowerBound(l_vec_);
    solver_.data()->setUpperBound(u_vec_);

    solver_.initSolver();

}

Eigen::VectorXd OptimizerFootTrajectory::solve()
{
    Eigen::Vector4d ctr;
    Eigen::VectorXd QPSolution;
    solver_.solveProblem();

    QPSolution = solver_.getSolution();
    return QPSolution;
    
}

void OptimizerFootTrajectory::integrate_trajectory_forward(Eigen::Vector3d foot_position, Eigen::Vector3d foot_velocity, Eigen::Vector3d foot_acceleration)
{
    for (int i = 0; i < 3; i++)
    {
        computed_foot_pos_(i) = foot_velocity(i) * dt_ + foot_position(i);
        computed_foot_vel_(i) = foot_acceleration(i) * dt_ + foot_velocity(i);
    }
}

Eigen::Vector3d OptimizerFootTrajectory::getComputedFootPos()
{
    return computed_foot_pos_;
}

Eigen::Vector3d OptimizerFootTrajectory::getComputedFootVel()
{
    return computed_foot_vel_;
}


int main()
{
    OptimizerFootTrajectory opt(0.01, 0.3);

    Eigen::Vector3d opt_weight_pos;
    opt_weight_pos << 55000, 55000, 5500000;
    Eigen::Vector3d opt_weight_vel;
    opt_weight_vel << 55000, 55000, 5500000;
    Eigen::Vector3d p_N_des;
    Eigen::Vector3d v_N_des;
    p_N_des << 0.2, 0.2, 0.0;
    v_N_des << 0, 0, 0;

    Eigen::Vector3d p_0_des;
    Eigen::Vector3d v_0_des;
    p_0_des << 0.0, 0.0, 0.0;
    v_0_des << 0.0, 0.0, 0.0;
    double v_max = 100;
    double a_max = 100;

    opt.update_P_and_q_matrices(opt_weight_pos, opt_weight_vel, p_N_des, v_N_des);
    opt.update_linear_matrix_and_bounds(p_0_des, v_0_des, v_max, a_max);
    opt.create_optimization_pb();

    Eigen::VectorXd sol;
    sol = opt.solve();

    std::vector<Eigen::Vector3d> foot_position;
    std::vector<Eigen::Vector3d> foot_velocity;
    std::vector<Eigen::Vector3d> foot_acceleration;
    // extract position, velocity and acceleration for x y z
    for (int i = 0; i < opt.nb_total_variables_per_coord_; i = i + 3)
    {
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
        Eigen::Vector3d acc;
        pos << sol(i), sol(i + opt.nb_total_variables_per_coord_), sol(i + 2 * opt.nb_total_variables_per_coord_);
        vel << sol(i + 1), sol(i + 1 + opt.nb_total_variables_per_coord_), sol(i + 1 + 2 * opt.nb_total_variables_per_coord_);
        acc << sol(i + 2), sol(i + 2 + opt.nb_total_variables_per_coord_), sol(i + 2 + 2 * opt.nb_total_variables_per_coord_);
        foot_position.push_back(pos);
        foot_velocity.push_back(vel);
        foot_acceleration.push_back(acc);
    }



    std::ofstream file("/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/capture_point/test/output.csv");
    file << "Position_X,Position_Y,Position_Z,Velocity_X,Velocity_Y,Velocity_Z,Acceleration_X,Acceleration_Y,Acceleration_Z\n";
    for (unsigned int i = 0; i < foot_position.size(); i++)
    {
        file << foot_position[i](0) << "," << foot_position[i](1) << "," << foot_position[i](2) << ","
            << foot_velocity[i](0) << "," << foot_velocity[i](1) << "," << foot_velocity[i](2) << ","
            << foot_acceleration[i](0) << "," << foot_acceleration[i](1) << "," << foot_acceleration[i](2) << "\n";
    }
    file.close();

    return 0;
}
