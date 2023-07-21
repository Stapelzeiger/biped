#include <string>
#include "foot_trajectory_optimization.hpp"
#include "gtest/gtest.h"

class OptimizerFootTrajectoryTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    double dt_ = 0.01;
    double Ts_ = 0.02;
    int N_ = static_cast<int>(Ts_/dt_);
    double nb_total_variables_per_coord_ = 3 * N_; // p, v, a
    double nb_total_variables_ = 3 * nb_total_variables_per_coord_; // x, y, z
    OptimizerFootTrajectory opt_ = OptimizerFootTrajectory(dt_, Ts_);
};


TEST_F(OptimizerFootTrajectoryTest, PMatrixQvecTest) {
    Eigen::Vector3d opt_weight_pos;
    opt_weight_pos << 1000, 1000, 1000;
    Eigen::Vector3d opt_weight_vel;
    opt_weight_vel << 500, 500, 500;
    Eigen::Vector3d final_pos;
    Eigen::Vector3d final_vel;
    final_pos << 0.5, 0.5, 0.5;
    final_vel << 0.0, 0.0, 0.0;
    Eigen::SparseMatrix<double> P_matrix_from_optimization;
    Eigen::VectorXd q_vec_from_optimization;

    opt_.get_P_and_q_matrices(opt_weight_pos, opt_weight_vel, final_pos, final_vel, P_matrix_from_optimization, q_vec_from_optimization);

    Eigen::SparseMatrix<double> P_matrix;
    Eigen::VectorXd q_vec;
    P_matrix.resize(nb_total_variables_, nb_total_variables_);
    q_vec.resize(nb_total_variables_, 1);
    q_vec.setZero();

    double factor_acc = 0.001;
    P_matrix.insert(2, 2) = factor_acc;
    P_matrix.insert(5, 5) = factor_acc;
    P_matrix.insert(8, 8) = factor_acc;
    P_matrix.insert(11, 11) = factor_acc;
    P_matrix.insert(14, 14) = factor_acc;
    P_matrix.insert(17, 17) = factor_acc;

    P_matrix.insert(3, 3) = opt_weight_pos(0);
    P_matrix.insert(4, 4) = opt_weight_vel(0);

    P_matrix.insert(9, 9) = opt_weight_pos(1);
    P_matrix.insert(10, 10) = opt_weight_vel(1);

    P_matrix.insert(15, 15) = opt_weight_pos(2);
    P_matrix.insert(16, 16) = opt_weight_vel(2);

    Eigen::MatrixXd dense_P_matrix = Eigen::MatrixXd(P_matrix);
    Eigen::MatrixXd dense_P_matrix_opt = Eigen::MatrixXd(P_matrix_from_optimization);

    q_vec(3) = - opt_weight_pos(0) * final_pos(0);
    q_vec(4) = - opt_weight_vel(0) * final_vel(0);
    q_vec(9) = - opt_weight_pos(1) * final_pos(1);
    q_vec(10) = - opt_weight_vel(1) * final_vel(1);
    q_vec(15) = - opt_weight_pos(2) * final_pos(2);
    q_vec(16) = - opt_weight_vel(2) * final_vel(2);

    EXPECT_TRUE(dense_P_matrix.isApprox(dense_P_matrix_opt));
    EXPECT_TRUE(q_vec.isApprox(q_vec_from_optimization));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
