#include <string>
#include "foot_trajectory.h"
#include "gtest/gtest.h"

TEST(TestValues, Testq_q_dot_qddot)
{
    SplineTrajectory spline;
    Eigen::Vector<double, 4> coeff = Eigen::Vector<double, 4>::Zero();
    coeff(1) = 1.0;
    double t = 2;
    auto q = spline.get_q(coeff, t);
    auto q_dot = spline.get_q_dot(coeff, t);
    auto q_ddot = spline.get_q_ddot(coeff, t);
    ASSERT_EQ(q, 2.0);
    ASSERT_EQ(q_dot, 1.0);
    ASSERT_EQ(q_ddot, 0.0);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}