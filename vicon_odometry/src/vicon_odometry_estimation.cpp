#include <memory>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cmath>
#include <list> 
#include <random>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include "eigen3/Eigen/Dense"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"


using namespace std::placeholders;
using namespace std::chrono_literals;


Eigen::Matrix3d cross_product_matrix(Eigen::Vector3d v)
{
  Eigen::Matrix3d v_cross;
  v_cross = Eigen::Matrix3d::Zero();
  v_cross(0, 1) = -v(2);
  v_cross(0, 2) = v(1);
  v_cross(1, 0) = v(2);
  v_cross(1, 2) = -v(0);
  v_cross(2, 0) = -v(1);
  v_cross(2, 1) = v(0);
  return v_cross;
}

/* Error state EKF https://arxiv.org/pdf/1711.02508.pdf */
class IMUPoseEKF
{
  public:
    IMUPoseEKF()
    {
      this->reset();
    }

    void reset(const Eigen::Quaterniond &att_0 = Eigen::Quaterniond::Identity())
    {
      // nominal states
      att_ = att_0;
      vel_I_ = Eigen::Vector3d::Zero();
      acc_bias_ = Eigen::Vector3d::Zero();
      gyro_bias_ = Eigen::Vector3d::Zero();
      // EKF states
      Eigen::Array<double, 1, 12> P_diag;
      P_diag << vel_std_0_, vel_std_0_, vel_std_0_,
          att_std_0_, att_std_0_, att_std_0_,
          acc_bias_std_0_, acc_bias_std_0_, acc_bias_std_0_,
          gyro_bias_std_0_, gyro_bias_std_0_, gyro_bias_std_0_;
      P_ = P_diag.pow(2.0).matrix().asDiagonal();
      // output states
      pos_ = Eigen::Vector3d::Zero();
      omega_ = Eigen::Vector3d::Zero();
    }

    void imu_update(const Eigen::Vector3d &acc_meas, const Eigen::Vector3d &angular_rate_meas, double dt)
    {
      omega_ = angular_rate_meas - gyro_bias_;
      Eigen::Vector3d acc = acc_meas - acc_bias_;
      auto rot_vec = omega_ * dt;
      Eigen::Quaterniond dq;
      dq.w() = 1;
      dq.vec() = rot_vec * 0.5;
      dq.normalize();

      auto R = att_.toRotationMatrix();
      Eigen::Matrix<double, 12, 12> F_x;
      F_x.setIdentity();
      F_x.block(0, 3, 3, 3) = -dt * R * cross_product_matrix(acc);
      F_x.block(0, 6, 3, 3) = -dt * R;
      F_x.block(3, 3, 3, 3) =  (dq.toRotationMatrix()).transpose();
      F_x.block(3, 9, 3, 3) = -dt * Eigen::MatrixXd::Identity(3, 3);

      Eigen::Matrix<double, 12, 12> Fi;
      Fi.setIdentity();

      Eigen::Matrix<double, 12, 12> Qi;
      Qi.setZero();
      double acc_noise_var = acc_noise_density_ * acc_noise_density_ / dt;
      double velocity_var = acc_noise_var * dt * dt;
      double gyro_noise_var = gyro_noise_density_ * gyro_noise_density_ / dt;
      double att_var = gyro_noise_var * dt * dt;
      double acc_bias_var = acc_bias_random_walk_ * acc_bias_random_walk_ * dt;
      double gyro_bias_var = gyro_bias_random_walk_ * gyro_bias_random_walk_ * dt;
      for (int i = 0; i < 3; i++)
      {
        Qi(i, i) = velocity_var;
        Qi(i + 3, i + 3) = att_var;
        Qi(i + 6, i + 6) = acc_bias_var;
        Qi(i + 9, i + 9) = gyro_bias_var;
      }
      
      // update nominal system
      att_ = att_ * dq;
      att_.normalize();
      const Eigen::Vector3d gravity_I(0, 0, -9.81);
      vel_I_ += dt * (R * acc + gravity_I);
      pos_ += dt * vel_I_;
      // update EKF (error state is zero)
      P_ = F_x * P_ * F_x.transpose() + Fi * Qi * Fi.transpose();
      // std::cout << "imu update: " << std::endl;
      // std::cout << P_ << std::endl;
      // std::cout << " att: " << att_.coeffs().transpose() << std::endl;
      // std::cout << " dq: " << rot_vec.transpose() << std::endl;
      // std::cout << " vel_I: " << vel_I_.transpose() << std::endl;
    }

    /* contact velocity, B = imu frame */
    Eigen::Vector3d compute_contact_residual(const Eigen::Vector3d &pos_B, const Eigen::Vector3d &vel_B) const
    {
      auto R = att_.toRotationMatrix();
      Eigen::Vector3d vel_B_total = vel_B + omega_.cross(pos_B);
      auto h = R * vel_B_total + vel_I_;
      return -h;
    }

    void get_state(Eigen::Vector3d &pos, Eigen::Vector3d &vel_I, Eigen::Quaterniond &att, Eigen::Vector3d &omega)
    {
      pos = pos_;
      att = att_;
      vel_I = vel_I_;
      omega = omega_;
    }

private:
  // nominal states
  Eigen::Quaterniond att_;
  Eigen::Vector3d vel_I_;
  Eigen::Vector3d acc_bias_;
  Eigen::Vector3d gyro_bias_;
  // EKF states
  Eigen::Matrix<double, 12, 12> P_; // [d_vel, d_att, d_acc_bias, d_gyro_bias]
  // output states
  Eigen::Vector3d pos_;
  Eigen::Vector3d omega_;

public:
  // noise parameters
  double acc_noise_density_;  // [m/s2 * sqrt(s)]
  double gyro_noise_density_;  // [rad/s * sqrt(s)]
  double acc_bias_random_walk_;  // [m/s2 / sqrt(s)]
  double gyro_bias_random_walk_;  // [rad/s / sqrt(s)]
  // initial covariance
  double vel_std_0_;
  double att_std_0_;
  double acc_bias_std_0_;
  double gyro_bias_std_0_;

};


class ViconOdometry : public rclcpp::Node
{
public:
  ViconOdometry()
  : Node("vicon_odometry")
  {

    base_link_frame_id_ = this->declare_parameter<std::string>("base_link_frame_id", "base_link");
    odom_frame_id_ = this->declare_parameter<std::string>("odom_frame_id", "odom");
    this->declare_parameter<bool>("publish_tf", false);

    // load noise parameters
    ekf_.vel_std_0_ = this->declare_parameter<double>("vel_std_0", 0.1);
    ekf_.att_std_0_ = this->declare_parameter<double>("att_std_0", 0.1);
    ekf_.acc_bias_std_0_ = this->declare_parameter<double>("acc_bias_std_0", 0.1);
    ekf_.gyro_bias_std_0_ = this->declare_parameter<double>("gyro_bias_std_0", 0.01);
    ekf_.acc_noise_density_ = this->declare_parameter<double>("acc_noise_density", 0.01);
    ekf_.gyro_noise_density_ = this->declare_parameter<double>("gyro_noise_density", 0.01);
    ekf_.acc_bias_random_walk_ = this->declare_parameter<double>("acc_bias_random_walk", 0.01);
    ekf_.gyro_bias_random_walk_ = this->declare_parameter<double>("gyro_bias_random_walk", 0.01);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("~/odom", 10);
    ekf_innovations_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/ekf_innovations", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "~/imu", 10, std::bind(&ViconOdometry::imu_cb, this, _1));

    reset_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "~/reset", 10, std::bind(&ViconOdometry::reset_cb, this, _1));

  }

private:

  void reset_cb(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (msg->data)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000 /* ms */, "Resetting estimator");
      ekf_.reset(q_IMU_to_BL_);
    }
  }

  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    // get imu transform (cached if the frame id is the same)
    if (msg->header.frame_id != imu_frame_id_)
    {
      if (msg->header.frame_id == "")
      {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000 /* ms */, "IMU message has no frame_id, skipping");
        return;
      }
      try {
        geometry_msgs::msg::TransformStamped T_IMU_to_BL = tf_buffer_->lookupTransform(base_link_frame_id_, msg->header.frame_id, rclcpp::Time(0));
        q_IMU_to_BL_ = Eigen::Quaterniond(T_IMU_to_BL.transform.rotation.w, T_IMU_to_BL.transform.rotation.x, T_IMU_to_BL.transform.rotation.y, T_IMU_to_BL.transform.rotation.z);
        p_IMU_in_BL_ << T_IMU_to_BL.transform.translation.x, T_IMU_to_BL.transform.translation.y, T_IMU_to_BL.transform.translation.z;
        ekf_.reset(q_IMU_to_BL_);
      } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "Could not get transform from %s to %s, skipping", msg->header.frame_id.c_str(), base_link_frame_id_.c_str());
        return;
      }
      RCLCPP_WARN_SKIPFIRST(this->get_logger(), "IMU frame id changed from %s to %s", msg->header.frame_id.c_str(), msg->header.frame_id.c_str());
      imu_frame_id_ = msg->header.frame_id;
    }

    // message time checks
    auto new_time = rclcpp::Time(msg->header.stamp);
    if (time_ == rclcpp::Time(0, 0, RCL_ROS_TIME))  // first message
    {
      time_ = new_time;
      return;
    }
    double dt = (new_time - time_).seconds();
    if (dt < 0)
    {
      RCLCPP_WARN(this->get_logger(), "IMU messages are not in order, skipping");
      return;
    }
    time_ = new_time;
    if (dt > 0.1)
    {
      RCLCPP_WARN(this->get_logger(), "IMU messages gap, resetting estimator");
      ekf_.reset(q_IMU_to_BL_);
      return;
    }

    // EKF time update
    Eigen::Vector3d acc_m;
    Eigen::Vector3d gyro_m;
    acc_m[0] = msg->linear_acceleration.x;
    acc_m[1] = msg->linear_acceleration.y;
    acc_m[2] = msg->linear_acceleration.z;
    gyro_m[0] = msg->angular_velocity.x;
    gyro_m[1] = msg->angular_velocity.y;
    gyro_m[2] = msg->angular_velocity.z;
    ekf_.imu_update(acc_m, gyro_m, dt);

    // get state estimate
    Eigen::Vector3d posIMU_I, velIMU_I, omegaIMU_IMU;
    Eigen::Quaterniond att_IMU_to_I;
    ekf_.get_state(posIMU_I, velIMU_I, att_IMU_to_I, omegaIMU_IMU);
    // transform from imu frame to base_link frame
    Eigen::Quaterniond att_BL_to_I = att_IMU_to_I * q_IMU_to_BL_.conjugate();
    Eigen::Vector3d posBL_I = posIMU_I - att_BL_to_I * p_IMU_in_BL_;
    Eigen::Vector3d omegaBL_BL = q_IMU_to_BL_ * omegaIMU_IMU;
    Eigen::Vector3d velIMU_BL = att_BL_to_I.conjugate() * velIMU_I;
    Eigen::Vector3d velBL_BL = velIMU_BL - omegaBL_BL.cross(p_IMU_in_BL_);
    // publish odometry
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = msg->header.stamp;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = base_link_frame_id_;
    odom.pose.pose.position.x = posBL_I[0];
    odom.pose.pose.position.y = posBL_I[1];
    odom.pose.pose.position.z = posBL_I[2];
    odom.pose.pose.orientation.w = att_BL_to_I.w();
    odom.pose.pose.orientation.x = att_BL_to_I.x();
    odom.pose.pose.orientation.y = att_BL_to_I.y();
    odom.pose.pose.orientation.z = att_BL_to_I.z();
    odom.twist.twist.linear.x = velBL_BL[0];
    odom.twist.twist.linear.y = velBL_BL[1];
    odom.twist.twist.linear.z = velBL_BL[2];
    odom.twist.twist.angular.x = omegaBL_BL[0];
    odom.twist.twist.angular.y = omegaBL_BL[1];
    odom.twist.twist.angular.z = omegaBL_BL[2];
    odom_pub_->publish(odom);


    if (this->get_parameter("publish_tf").as_bool())
    {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = msg->header.stamp;
      tf.header.frame_id = odom_frame_id_;
      tf.child_frame_id = base_link_frame_id_;
      tf.transform.translation.x = posBL_I[0];
      tf.transform.translation.y = posBL_I[1];
      tf.transform.translation.z = posBL_I[2];
      tf.transform.rotation.w = att_BL_to_I.w();
      tf.transform.rotation.x = att_BL_to_I.x();
      tf.transform.rotation.y = att_BL_to_I.y();
      tf.transform.rotation.z = att_BL_to_I.z();
      tf_broadcaster_->sendTransform(tf);
    }
  }

  rclcpp::Time time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  IMUPoseEKF ekf_;

  std::string imu_frame_id_;
  std::string base_link_frame_id_;
  std::string odom_frame_id_;
  Eigen::Quaterniond q_IMU_to_BL_;
  Eigen::Vector3d p_IMU_in_BL_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reset_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ekf_innovations_marker_pub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ViconOdometry>());
  rclcpp::shutdown();
  return 0;
}
