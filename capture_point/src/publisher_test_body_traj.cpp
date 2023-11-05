#include <memory>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cmath>
#include <list>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "rosgraph_msgs/msg/clock.hpp"
#include "eigen3/Eigen/Dense"
#include "foot_trajectory.h"

using namespace std::placeholders;
using namespace std::chrono_literals;

class TestTrajFollowing : public rclcpp::Node
{
public:
    TestTrajFollowing();
    ~TestTrajFollowing(){};

    void publish_body_trajectories(std::string frame_id, Eigen::Vector3d pos_body, Eigen::Quaterniond quat_body, Eigen::Vector3d vel_body, Eigen::Vector3d acc_body,
                                Eigen::Vector3d pos_right_foot, Eigen::Quaterniond quat_right_foot, Eigen::Vector3d vel_right_foot, Eigen::Vector3d acc_right_foot,
                                Eigen::Vector3d pos_left_foot, Eigen::Quaterniond quat_left_foot, Eigen::Vector3d vel_left_foot, Eigen::Vector3d acc_left_foot);
    void timer_callback();

private:
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_body_trajectory_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string r_foot_urdf_frame_id_;
    std::string l_foot_urdf_frame_id_;
    std::string base_link_frame_id_;
    std::string swing_foot_name_;

    FootTrajectory spline_traj_swing_foot_;
    Eigen::Vector3d current_des_swing_foot_STF_;
    Eigen::Vector3d setpt_pos_swing_foot_STF_;
    Eigen::Vector3d setpt_vel_swing_foot_STF_;
    Eigen::Vector3d next_des_swing_foot_STF_;
    Eigen::Vector3d current_des_stance_foot_STF_;

    double dt_;
    double duration_traj_;
    double time_;
    double robot_height_;
};

TestTrajFollowing::TestTrajFollowing() : Node("pub_traj_bodies")
{
    r_foot_urdf_frame_id_ = this->declare_parameter<std::string>("r_foot_urdf_frame_id", "R_FOOT");
    l_foot_urdf_frame_id_ = this->declare_parameter<std::string>("l_foot_urdf_frame_id", "L_FOOT");
    base_link_frame_id_ = this->declare_parameter<std::string>("base_link_frame_id", "base_link");
    robot_height_ = this->declare_parameter<double>("robot_height",  0.54);

    pub_body_trajectory_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/body_trajectories", 10);

    current_des_swing_foot_STF_ = Eigen::Vector3d(0.0, 0.2, 0.2);
    // current_des_stance_foot_STF_ = Eigen::Vector3d(0.0, 0.2, 0.0);
    next_des_swing_foot_STF_ = Eigen::Vector3d(0.0, 0.3, 0.05);
    current_des_stance_foot_STF_ = Eigen::Vector3d(0.0, 0.0, 0.0);

    swing_foot_name_ = r_foot_urdf_frame_id_;

    dt_ = 0.01;
    duration_traj_ = 0.5;
    time_ = 0.0;

    spline_traj_swing_foot_ = FootTrajectory(duration_traj_, dt_);
    spline_traj_swing_foot_.set_desired_end_position(next_des_swing_foot_STF_);
    spline_traj_swing_foot_.set_initial_position(current_des_swing_foot_STF_);

    std::chrono::duration<double> period = dt_ * 1s;
    timer_ = rclcpp::create_timer(this, this->get_clock(), period, std::bind(&TestTrajFollowing::timer_callback, this));
}

void TestTrajFollowing::timer_callback()
{
    std::cout << "r_foot_urdf_frame_id_" << r_foot_urdf_frame_id_ << std::endl;
    std::cout << "l_foot_urdf_frame_id_" << l_foot_urdf_frame_id_ << std::endl;
    Eigen::Vector3d pos_body_level_STF = Eigen::Vector3d(0.03, 0.0, robot_height_);
    std::string frame_id = "odom";

    Eigen::Vector3d pos_desired_swing_foot_STF, vel_desired_swing_foot_STF, acc_desired_swing_foot_STF;

    spline_traj_swing_foot_.get_traj_foot_pos_vel(time_, pos_desired_swing_foot_STF, vel_desired_swing_foot_STF, acc_desired_swing_foot_STF);

    std::cout << vel_desired_swing_foot_STF.transpose() << std::endl;
    pos_desired_swing_foot_STF(2) = 0.0;
    vel_desired_swing_foot_STF(2) = 0.0;

    double R_circle_traj = 0.05;
    double omega_circle_traj = 2.0 * M_PI / duration_traj_;
    double x_circle_traj = R_circle_traj * cos(omega_circle_traj * time_);
    double vel_x_circle_traj = -R_circle_traj * omega_circle_traj * sin(omega_circle_traj * time_);
    double acc_x_circle_traj = -R_circle_traj * omega_circle_traj * omega_circle_traj * cos(omega_circle_traj * time_);
    double y_circle_traj = R_circle_traj * sin(omega_circle_traj * time_) - 0.1;
    double vel_y_circle_traj = R_circle_traj * omega_circle_traj * cos(omega_circle_traj * time_);
    double acc_y_circle_traj = -R_circle_traj * omega_circle_traj * omega_circle_traj * sin(omega_circle_traj * time_);
    double z_circle_traj = 0.0;

    pos_desired_swing_foot_STF << x_circle_traj, y_circle_traj, 0.0;
    vel_desired_swing_foot_STF << vel_x_circle_traj, vel_y_circle_traj, 0.0;
    acc_desired_swing_foot_STF << acc_x_circle_traj, acc_y_circle_traj, 0.0;


    pos_desired_swing_foot_STF = current_des_swing_foot_STF_;
    vel_desired_swing_foot_STF = Eigen::Vector3d::Zero();
    acc_desired_swing_foot_STF = Eigen::Vector3d::Zero();

    publish_body_trajectories(frame_id, pos_body_level_STF, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                              current_des_stance_foot_STF_, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                              pos_desired_swing_foot_STF, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), vel_desired_swing_foot_STF, acc_desired_swing_foot_STF);
    


    time_ += dt_;
    if (time_ >= duration_traj_)
    {
        // current_des_stance_foot_STF_ = current_des_swing_foot_STF_;
        // std::swap(current_des_swing_foot_STF_, next_des_swing_foot_STF_);
        time_ = 0.0;
        spline_traj_swing_foot_.set_initial_position(current_des_swing_foot_STF_);
        spline_traj_swing_foot_.set_desired_end_position(next_des_swing_foot_STF_);

        if (swing_foot_name_ == r_foot_urdf_frame_id_)
        {
            swing_foot_name_ = l_foot_urdf_frame_id_;
        }
        else
        {
            swing_foot_name_ = r_foot_urdf_frame_id_;
        }
        std::cout << swing_foot_name_ << std::endl;

    }
}

void TestTrajFollowing::publish_body_trajectories(std::string frame_id, Eigen::Vector3d pos_body, Eigen::Quaterniond quat_body, Eigen::Vector3d vel_body, Eigen::Vector3d acc_body,
                                Eigen::Vector3d pos_right_foot, Eigen::Quaterniond quat_right_foot, Eigen::Vector3d vel_right_foot, Eigen::Vector3d acc_right_foot,
                                Eigen::Vector3d pos_left_foot, Eigen::Quaterniond quat_left_foot, Eigen::Vector3d vel_left_foot, Eigen::Vector3d acc_left_foot)
{
    trajectory_msgs::msg::MultiDOFJointTrajectory msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id;
    msg.joint_names.push_back(base_link_frame_id_);
    msg.joint_names.push_back(r_foot_urdf_frame_id_);
    msg.joint_names.push_back(l_foot_urdf_frame_id_);

    geometry_msgs::msg::Transform body;
    body.translation.x = pos_body(0);
    body.translation.y = pos_body(1);
    body.translation.z = pos_body(2);
    body.rotation.x = quat_body.x();
    body.rotation.y = quat_body.y();
    body.rotation.z = quat_body.z();
    body.rotation.w = quat_body.w();

    geometry_msgs::msg::Twist body_vel;
    body_vel.linear.x = vel_body(0);
    body_vel.linear.y = vel_body(1);
    body_vel.linear.z = vel_body(2);
    geometry_msgs::msg::Twist body_acc;
    body_acc.linear.x = acc_body(0);
    body_acc.linear.y = acc_body(1);
    body_acc.linear.z = acc_body(2);

    geometry_msgs::msg::Transform right_foot;
    right_foot.translation.x = pos_right_foot(0);
    right_foot.translation.y = pos_right_foot(1);
    right_foot.translation.z = pos_right_foot(2);
    right_foot.rotation.x = quat_right_foot.x();
    right_foot.rotation.y = quat_right_foot.y();
    right_foot.rotation.z = quat_right_foot.z();
    right_foot.rotation.w = quat_right_foot.w();

    geometry_msgs::msg::Twist right_foot_vel;
    right_foot_vel.linear.x = vel_right_foot(0);
    right_foot_vel.linear.y = vel_right_foot(1);
    right_foot_vel.linear.z = vel_right_foot(2);
    geometry_msgs::msg::Twist right_foot_acc;
    right_foot_acc.linear.x = acc_right_foot(0);
    right_foot_acc.linear.y = acc_right_foot(1);
    right_foot_acc.linear.z = acc_right_foot(2);

    geometry_msgs::msg::Transform left_foot;
    left_foot.translation.x = pos_left_foot(0);
    left_foot.translation.y = pos_left_foot(1);
    left_foot.translation.z = pos_left_foot(2);
    left_foot.rotation.x = quat_left_foot.x();
    left_foot.rotation.y = quat_left_foot.y();
    left_foot.rotation.z = quat_left_foot.z();
    left_foot.rotation.w = quat_left_foot.w();

    geometry_msgs::msg::Twist left_foot_vel;
    left_foot_vel.linear.x = vel_left_foot(0);
    left_foot_vel.linear.y = vel_left_foot(1);
    left_foot_vel.linear.z = vel_left_foot(2);
    geometry_msgs::msg::Twist left_foot_acc;
    left_foot_acc.linear.x = acc_left_foot(0);
    left_foot_acc.linear.y = acc_left_foot(1);
    left_foot_acc.linear.z = acc_left_foot(2);

    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint body_traj_point;
    body_traj_point.transforms.push_back(body);
    body_traj_point.velocities.push_back(body_vel);
    body_traj_point.accelerations.push_back(body_acc);

    body_traj_point.transforms.push_back(right_foot);
    body_traj_point.velocities.push_back(right_foot_vel);
    body_traj_point.accelerations.push_back(right_foot_acc);
    body_traj_point.transforms.push_back(left_foot);
    body_traj_point.velocities.push_back(left_foot_vel);
    body_traj_point.accelerations.push_back(left_foot_acc);

    msg.points.push_back(body_traj_point);
    pub_body_trajectory_->publish(msg);
}


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TestTrajFollowing>());
    rclcpp::shutdown();
    return 0;
}