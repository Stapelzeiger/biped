#include <string>
#include "foot_trajectory_optimization.hpp"
#include "gtest/gtest.h"


TEST(OptimizerFootTrajectoryTest, IntegrateTrajectoryForwardTest) {
    double dt = 0.1;
    double Ts = 1.0;
    OptimizerFootTrajectory opt(dt, Ts);

    Eigen::Vector3d foot_position = 2 * Eigen::Vector3d::Ones();
    Eigen::Vector3d foot_velocity = Eigen::Vector3d::Ones();
    Eigen::Vector3d foot_acceleration = Eigen::Vector3d::Constant(2.0);

    opt.integrate_trajectory_forward(foot_position, foot_velocity, foot_acceleration);

    Eigen::Vector3d expected_foot_position = 2* Eigen::Vector3d::Ones() + Eigen::Vector3d::Ones() * dt; // v*dt + p for each dimension
    Eigen::Vector3d expected_foot_velocity = Eigen::Vector3d::Constant(2.0)*dt + Eigen::Vector3d::Ones(); // a*dt + v for each dimension

    EXPECT_TRUE(expected_foot_position.isApprox(opt.getComputedFootPos()));
    EXPECT_TRUE(expected_foot_velocity.isApprox(opt.getComputedFootVel()));
}


TEST(OptimizerFootTrajectoryTest, PMatrixQvecTest) {
    double dt = 0.01;
    double Ts = 0.02;
    int N = static_cast<int>(Ts/ dt);
    OptimizerFootTrajectory opt(dt, Ts);
    double nb_total_variables_per_coord = 3 * N; // p, v, a
    double nb_total_variables = 3 * nb_total_variables_per_coord; // x, y, z

    double opt_weight_pos = 1000;
    double opt_weight_vel = 500;
    Eigen::Vector3d p_N_des;
    Eigen::Vector3d v_N_des;
    p_N_des << 0.5, 0.5, 0.5;
    v_N_des << 0.0, 0.0, 0.0;

    bool status_update_P_q_matrices = opt.update_P_and_q_matrices(opt_weight_pos, opt_weight_vel, p_N_des, v_N_des);

    Eigen::SparseMatrix<double> P_matrix;
    Eigen::VectorXd q_vec;
    P_matrix.resize(nb_total_variables, nb_total_variables);
    q_vec.resize(nb_total_variables, 1);
    q_vec.setZero();

    P_matrix.insert(2, 2) = 1;
    P_matrix.insert(5, 5) = 1;
    P_matrix.insert(8, 8) = 1;
    P_matrix.insert(11, 11) = 1;
    P_matrix.insert(14, 14) = 1;
    P_matrix.insert(17, 17) = 1;

    P_matrix.insert(3, 3) = opt_weight_pos;
    P_matrix.insert(4, 4) = opt_weight_vel;
    P_matrix.insert(9, 9) = opt_weight_pos;
    P_matrix.insert(10, 10) = opt_weight_vel;
    P_matrix.insert(15, 15) = opt_weight_pos;
    P_matrix.insert(16, 16) = opt_weight_vel;

    Eigen::MatrixXd dense_P_matrix = Eigen::MatrixXd(P_matrix);
    Eigen::MatrixXd dense_P_matrix_opt = Eigen::MatrixXd(opt.P_matrix_);

    q_vec(3) = -2 * opt_weight_pos * p_N_des(0);
    q_vec(4) = -2 * opt_weight_vel * v_N_des(0);
    q_vec(9) = -2 * opt_weight_pos * p_N_des(1);
    q_vec(10) = -2 * opt_weight_vel * v_N_des(1);
    q_vec(15) = -2 * opt_weight_pos * p_N_des(2);
    q_vec(16) = -2 * opt_weight_vel * v_N_des(2);

    std::cout << q_vec << std::endl;
    std::cout << opt.q_vec_ << std::endl;

    EXPECT_TRUE(dense_P_matrix.isApprox(dense_P_matrix_opt));
    EXPECT_TRUE(q_vec.isApprox(opt.q_vec_));
    EXPECT_TRUE(status_update_P_q_matrices);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}