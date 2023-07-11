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

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}