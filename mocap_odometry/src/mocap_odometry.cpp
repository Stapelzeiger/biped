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

struct IMUData {
    sensor_msgs::msg::Imu imu_msg;
    rclcpp::Time timestamp;

    IMUData(sensor_msgs::msg::Imu msg, rclcpp::Time ts)
        : imu_msg(msg), timestamp(ts) {}

    bool operator<(const IMUData& other) const {
        return timestamp < other.timestamp;
    }
};

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

    void reset(const Eigen::Quaterniond &att_0 = Eigen::Quaterniond::Identity(), const Eigen::Vector3d &pos_0 = Eigen::Vector3d::Zero())
    {
      // nominal states
      att_ = att_0;
      pos_I_ = pos_0;
      vel_I_ = Eigen::Vector3d::Zero();
      acc_bias_ = Eigen::Vector3d::Zero();
      gyro_bias_ = Eigen::Vector3d::Zero();
      // EKF states
      Eigen::Array<double, 1, NX> P_diag;
      P_diag.setZero();
      for (int i = 0; i < 3; i++)
      {
        P_diag(XVEL + i) = vel_std_0_;
        P_diag(XATT + i) = att_std_0_;
        P_diag(XACC_BIAS + i) = acc_bias_std_0_;
        P_diag(XGYRO_BIAS + i) = gyro_bias_std_0_;
        P_diag(XPOS + i) = pos_std_0_;
      }
      P_ = P_diag.pow(2.0).matrix().asDiagonal();

      // output states
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
      Eigen::Matrix<double, NX, NX> F_x;
      F_x.setIdentity();
      F_x.block(XVEL, XATT, 3, 3) = -dt * R * cross_product_matrix(acc);
      F_x.block(XVEL, XACC_BIAS, 3, 3) = -dt * R;
      F_x.block(XATT, XATT, 3, 3) =  (dq.toRotationMatrix()).transpose();
      F_x.block(XATT, XGYRO_BIAS, 3, 3) = -dt * Eigen::MatrixXd::Identity(3, 3);
      F_x.block(XPOS, XVEL, 3, 3) = dt * Eigen::MatrixXd::Identity(3, 3);

      Eigen::Matrix<double, NX, NX> Fi;
      Fi.setIdentity();

      Eigen::Matrix<double, NX, NX> Qi;
      Qi.setZero();
      double acc_noise_var = acc_noise_density_ * acc_noise_density_ / dt;
      double velocity_var = acc_noise_var * dt * dt;
      double gyro_noise_var = gyro_noise_density_ * gyro_noise_density_ / dt;
      double att_var = gyro_noise_var * dt * dt;
      double acc_bias_var = acc_bias_random_walk_ * acc_bias_random_walk_ * dt;
      double gyro_bias_var = gyro_bias_random_walk_ * gyro_bias_random_walk_ * dt;
      for (int i = 0; i < 3; i++)
      {
        Qi(XVEL + i, XVEL + i) = velocity_var;
        Qi(XATT + i, XATT + i) = att_var;
        Qi(XACC_BIAS + i, XACC_BIAS + i) = acc_bias_var;
        Qi(XGYRO_BIAS + i, XGYRO_BIAS + i) = gyro_bias_var;
        Qi(XPOS + i, XPOS + i) = 0.0;
      }

      // update nominal system
      att_ = att_ * dq;
      att_.normalize();
      const Eigen::Vector3d gravity_I(0, 0, -9.81);
      vel_I_ += dt * (R * acc + gravity_I);
      pos_I_ += dt * vel_I_;
      // update EKF (error state is zero)
      P_ = F_x * P_ * F_x.transpose() + Fi * Qi * Fi.transpose();
      // std::cout << "imu update: " << std::endl;
      // std::cout << P_ << std::endl;
      // std::cout << " att: " << att_.coeffs().transpose() << std::endl;
      // std::cout << " dq: " << rot_vec.transpose() << std::endl;
      // std::cout << " vel_I: " << vel_I_.transpose() << std::endl;
    }

  void pos_measurement(const Eigen::Vector3d &pos_B, const Eigen::Vector3d &pos_I_meas, const Eigen::Matrix3d &R_cov,
      Eigen::Vector3d &h_out, Eigen::Matrix3d &h_cov_out)
    {
      // h(x) = R * pos_B + pos_I
      auto R = att_.toRotationMatrix();
      Eigen::Vector3d h = R * pos_B + pos_I_;
      // h = (R * R(delta_att) * pos_B) + pos_I = R * (I + S(delta_att)) * pos_B + pos_I + O(|delta_att|^2)
      Eigen::Matrix<double, 3, NX> H;
      H.setZero();
      // partial h / partial delta_vel_I = [0, 0, 0]
      // partial h / partial delta_att = -R * S(pos_B)
      H.block(0, XATT, 3, 3) = -R * cross_product_matrix(pos_B);
      // partial h / partial delta_acc_bias = [0, 0, 0]
      // partial h / partial delta_gyro_bias = [0, 0, 0]
      // partial h / partial delta_pos_I = I
      H.block(0, XPOS, 3, 3) = Eigen::MatrixXd::Identity(3, 3);

      auto h_cov = H * P_ * H.transpose() + R_cov;
      auto K = P_ * H.transpose() * h_cov.inverse();
      Eigen::Matrix<double, NX, 1> delta_x = K * (pos_I_meas - h);
      P_ = (Eigen::MatrixXd::Identity(NX, NX) - K * H) * P_;

      // update nominal system
      pos_I_ += delta_x.segment<3>(XPOS);
      vel_I_ += delta_x.segment<3>(XVEL);
      Eigen::Quaterniond delta_att;
      delta_att.vec() = delta_x.block(XATT, 0, 3, 1) * 0.5;
      delta_att.w() = 1;
      att_ = att_ * delta_att;
      att_.normalize();
      acc_bias_ += delta_x.segment<3>(XACC_BIAS);
      gyro_bias_ += delta_x.segment<3>(XGYRO_BIAS);

      // debug outputs
      h_out = h;
      h_cov_out = h_cov;
    }

    void att_measurement(const Eigen::Quaterniond &quat_meas, const Eigen::Matrix3d &R_cov, Eigen::Vector3d &h_out, Eigen::Matrix3d &h_cov_out)
    {
      // h(x) = delta_att
      Eigen::Vector3d h = Eigen::Vector3d::Zero();
      Eigen::Matrix<double, 3, NX> H;
      H.setZero();
      // partial h / partial delta_vel_I = [0, 0, 0]
      // partial h / partial delta_att = 
      H.block(0, XATT, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
      // partial h / partial delta_acc_bias = [0, 0, 0]
      // partial h / partial delta_gyro_bias = [0, 0, 0]
      // partial h / partial delta_pos_I = [0, 0, 0]

      auto q_nom_to_I = att_;
      auto q_meas_to_I = quat_meas;
      auto q_meas_to_nom = q_nom_to_I.conjugate() * q_meas_to_I;
      Eigen::Vector3d y;
      if (q_meas_to_nom.w() > 0) {
        y = 2 * q_meas_to_nom.vec();
      } else {
        y = -2 * q_meas_to_nom.vec();
      }
      auto h_cov = H * P_ * H.transpose() + R_cov;
      auto K = P_ * H.transpose() * h_cov.inverse();
      Eigen::Matrix<double, NX, 1> delta_x = K * (y - h);
      P_ = (Eigen::MatrixXd::Identity(NX, NX) - K * H) * P_;

      // update nominal system
      pos_I_ += delta_x.segment<3>(XPOS);
      vel_I_ += delta_x.segment<3>(XVEL);
      Eigen::Quaterniond delta_att;
      delta_att.vec() = delta_x.block(XATT, 0, 3, 1) * 0.5;
      delta_att.w() = 1;
      att_ = att_ * delta_att;
      att_.normalize();
      acc_bias_ += delta_x.segment<3>(XACC_BIAS);
      gyro_bias_ += delta_x.segment<3>(XGYRO_BIAS);

      // debug outputs
      h_out = h;
      h_cov_out = h_cov;
    }

    void get_state(Eigen::Vector3d &pos_I, Eigen::Vector3d &vel_I, Eigen::Quaterniond &att, Eigen::Vector3d &omega)
    {
      vel_I = vel_I_;
      att = att_;
      omega = omega_;
      pos_I = pos_I_;
    }

private:
  // nominal states
  Eigen::Quaterniond att_;
  Eigen::Vector3d pos_I_;
  Eigen::Vector3d vel_I_;
  Eigen::Vector3d acc_bias_;
  Eigen::Vector3d gyro_bias_;
  // EKF states
  static const int NX = 15;
  static const int XVEL = 0;
  static const int XATT = 3;
  static const int XACC_BIAS = 6;
  static const int XGYRO_BIAS = 9;
  static const int XPOS = 12;
  Eigen::Matrix<double, NX, NX> P_; // [d_vel, d_att, d_acc_bias, d_gyro_bias, d_pos]
  // output states
  Eigen::Vector3d omega_;

public:
  // noise parameters
  double acc_noise_density_;  // [m/s2 * sqrt(s)]
  double gyro_noise_density_;  // [rad/s * sqrt(s)]
  double acc_bias_random_walk_;  // [m/s2 / sqrt(s)]
  double gyro_bias_random_walk_;  // [rad/s / sqrt(s)]
  // initial covariance
  double pos_std_0_;
  double vel_std_0_;
  double att_std_0_;
  double acc_bias_std_0_;
  double gyro_bias_std_0_;

};


class MocapOdometry : public rclcpp::Node
{
public:
  MocapOdometry()
  : Node("mocap_odometry")
  {

    base_link_frame_id_ = this->declare_parameter<std::string>("base_link_frame_id", "base_link");
    base_link_mocap_frame_id_ = this->declare_parameter<std::string>("base_link_mocap_frame_id", "base_link_mocap");
    odom_frame_id_ = this->declare_parameter<std::string>("odom_frame_id", "world");
    this->declare_parameter<bool>("publish_tf", false);

    // load noise parameters
    ekf_.pos_std_0_ = this->declare_parameter<double>("pos_std_0", 0.1);
    ekf_.vel_std_0_ = this->declare_parameter<double>("vel_std_0", 0.1);
    ekf_.att_std_0_ = this->declare_parameter<double>("att_std_0", 0.1);
    ekf_.acc_bias_std_0_ = this->declare_parameter<double>("acc_bias_std_0", 0.1);
    ekf_.gyro_bias_std_0_ = this->declare_parameter<double>("gyro_bias_std_0", 0.01);
    ekf_.acc_noise_density_ = this->declare_parameter<double>("acc_noise_density", 0.01);
    ekf_.gyro_noise_density_ = this->declare_parameter<double>("gyro_noise_density", 0.01);
    ekf_.acc_bias_random_walk_ = this->declare_parameter<double>("acc_bias_random_walk", 0.01);
    ekf_.gyro_bias_random_walk_ = this->declare_parameter<double>("gyro_bias_random_walk", 0.01);
    mocap_pos_std_ = this->declare_parameter<double>("mocap_pos_std", 0.1);
    mocap_att_std_ = this->declare_parameter<double>("mocap_att_std", 0.1);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("~/odom", 10);
    ekf_innovations_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/ekf_innovations", 10);
    
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "~/imu", 10, std::bind(&MocapOdometry::imu_cb, this, _1));

    reset_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "~/reset", 10, std::bind(&MocapOdometry::reset_cb, this, _1));

    mocap_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/mocap/pose", 10, std::bind(&MocapOdometry::mocap_cb, this, _1));

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

  void mocap_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    // get mocap transform (cached if the frame id is the same)
    if (msg->header.frame_id != mocap_frame_id_)
    {
      if (msg->header.frame_id == "")
      {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000 /* ms */, "mocap message has no frame_id, skipping");
        return;
      }
      try {
        geometry_msgs::msg::TransformStamped T_MOCAP_to_I = tf_buffer_->lookupTransform(odom_frame_id_, msg->header.frame_id, rclcpp::Time(0));
        T_MOCAP_to_I_.translation() << T_MOCAP_to_I.transform.translation.x, T_MOCAP_to_I.transform.translation.y, T_MOCAP_to_I.transform.translation.z;
        T_MOCAP_to_I_.linear() = Eigen::Quaterniond(T_MOCAP_to_I.transform.rotation.w,
                                            T_MOCAP_to_I.transform.rotation.x,
                                            T_MOCAP_to_I.transform.rotation.y,
                                            T_MOCAP_to_I.transform.rotation.z).toRotationMatrix();
      } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "Could not get transform from %s to %s, skipping", msg->header.frame_id.c_str(), imu_frame_id_.c_str());
        return;
      }
      RCLCPP_WARN_SKIPFIRST(this->get_logger(), "Mocap frame id changed from %s to %s", msg->header.frame_id.c_str(), msg->header.frame_id.c_str());
      mocap_frame_id_ = msg->header.frame_id;
    }
    // TODO: do not run this everytime.
    try {
        geometry_msgs::msg::TransformStamped T_TRACKER_to_IMU = tf_buffer_->lookupTransform(imu_frame_id_, base_link_mocap_frame_id_, rclcpp::Time(0));
        T_TRACKER_to_IMU_.translation() << T_TRACKER_to_IMU.transform.translation.x, T_TRACKER_to_IMU.transform.translation.y, T_TRACKER_to_IMU.transform.translation.z;
        T_TRACKER_to_IMU_.linear() = Eigen::Quaterniond(T_TRACKER_to_IMU.transform.rotation.w,
                                            T_TRACKER_to_IMU.transform.rotation.x,
                                            T_TRACKER_to_IMU.transform.rotation.y,
                                            T_TRACKER_to_IMU.transform.rotation.z).toRotationMatrix();

      } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "Could not get transform from %s to %s, skipping", base_link_mocap_frame_id_.c_str(), imu_frame_id_.c_str());
        return;
    }

    Eigen::Transform<double, 3, Eigen::Isometry> T_TRACKER_to_MOCAP;
    T_TRACKER_to_MOCAP.translation() << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    T_TRACKER_to_MOCAP.linear() = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z).toRotationMatrix();

    const auto T_TRACKER_to_I = T_MOCAP_to_I_ * T_TRACKER_to_MOCAP;
    const auto p_TRACKER_in_I = T_TRACKER_to_I.translation();
    const auto p_TRACKER_in_IMU = T_TRACKER_to_IMU_.translation();

    // update EKF for position & att. measurement
    Eigen::Matrix3d R_cov = Eigen::Array3d(mocap_pos_std_, mocap_pos_std_, mocap_pos_std_).pow(2.0).matrix().asDiagonal();
    Eigen::Vector3d h_out; // todo publish for plotting
    Eigen::Matrix3d h_cov_out;

    Eigen::Matrix3d R_cov_att = Eigen::Array3d(mocap_att_std_, mocap_att_std_, mocap_att_std_).pow(2.0).matrix().asDiagonal();
    Eigen::Vector3d h_out_att; // todo publish for plotting
    Eigen::Matrix3d h_cov_out_att;
    const auto T_IMU_to_I = T_TRACKER_to_I * T_TRACKER_to_IMU_.inverse();
    const auto q_IMU_to_I = Eigen::Quaterniond(T_IMU_to_I.linear());

    // message time checks
    auto new_time = rclcpp::Time(msg->header.stamp);
    if (last_mocap_time_ == rclcpp::Time(0, 0, RCL_ROS_TIME)) {  // first message
      last_mocap_time_ = new_time;
      ekf_.reset(q_IMU_to_I, T_IMU_to_I.translation());
      return;
    }
    double dt = (new_time - last_mocap_time_).seconds();
    if (dt < 0) {
      RCLCPP_WARN(this->get_logger(), "Mocap messages are not in order, skipping");
      return;
    }

    last_mocap_time_ = new_time;
    if (dt > 0.1){
      RCLCPP_WARN(this->get_logger(), "Mocap messages gap");
      ekf_.reset(q_IMU_to_I, T_IMU_to_I.translation());
      return;
    }

    ekf_.pos_measurement(p_TRACKER_in_IMU, p_TRACKER_in_I, R_cov, h_out, h_cov_out);
    ekf_.att_measurement(q_IMU_to_I, R_cov_att, h_out_att, h_cov_out_att);
  }


  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "IMU message received");

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
        q_IMU_to_BL_ = Eigen::Quaterniond(T_IMU_to_BL.transform.rotation.w,
                                          T_IMU_to_BL.transform.rotation.x,
                                          T_IMU_to_BL.transform.rotation.y,
                                          T_IMU_to_BL.transform.rotation.z);
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
  rclcpp::Time last_mocap_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  IMUPoseEKF ekf_;

  std::string imu_frame_id_;
  std::string base_link_frame_id_;
  std::string base_link_mocap_frame_id_;
  std::string odom_frame_id_;
  std::string mocap_frame_id_;
  Eigen::Quaterniond q_IMU_to_BL_;
  Eigen::Vector3d p_IMU_in_BL_;
  Eigen::Vector3d p_TRACKER_in_IMU_;
  Eigen::Quaterniond q_TRACKER_to_IMU_;
  Eigen::Quaterniond q_MOCAP_to_I_;
  Eigen::Vector3d p_MOCAP_in_I_;

  double mocap_pos_std_;
  double mocap_att_std_;

  Eigen::Transform<double, 3, Eigen::Isometry> T_TRACKER_to_IMU_;
  Eigen::Transform<double, 3, Eigen::Isometry> T_MOCAP_to_I_;

  bool publish_tf_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reset_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mocap_sub_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ekf_innovations_marker_pub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MocapOdometry>());
  rclcpp::shutdown();
  return 0;
}
