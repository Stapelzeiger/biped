#include <memory>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cmath>
#include <list> 
#include <random>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "std_msgs/msg/string.hpp"
#include "biped_bringup/msg/stamped_bool.hpp"
#include "eigen3/Eigen/Dense"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#pragma GCC diagnostic pop

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
      std::cout << "imu update: " << std::endl;
      std::cout << P_ << std::endl;
      // std::cout << " att: " << att_.coeffs().transpose() << std::endl;
      // std::cout << " dq: " << rot_vec.transpose() << std::endl;
      // std::cout << " vel_I: " << vel_I_.transpose() << std::endl;
    }

    /* measures a contact zero velocity in imu frame */
    void zero_contact_vel_measurement_update(const Eigen::Vector3d &pos_B, const Eigen::Vector3d &vel_B,
      const Eigen::Matrix3d &vel_B_noise_cov, Eigen::Vector3d &h_out, Eigen::Matrix3d &h_cov_out)
    {
      // h(x) = R (v_B + omega_m x pos_B - gyro_bias x pos_B) + v_I
      auto R = att_.toRotationMatrix();
      Eigen::Vector3d vel_B_total = vel_B + omega_.cross(pos_B);
      auto h = R * vel_B_total + vel_I_;
      Eigen::Matrix<double, 3, 12> H;
      H.setZero();
      H.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);  // partial h / partial delta_vel_I
      // h = R * R(delta_att) * V + v_I  =  R * (I + S(delta_att)) * V + v_I + O(|delta_att|^2)
      // => partial h / partial delta_att = -R * S(V)
      H.block(0, 3, 3, 3) = -R * cross_product_matrix(vel_B_total);  // partial h / partial delta_att
      // partial h / partial accel_bias = 0
      H.block(0, 9, 3, 3) = R * cross_product_matrix(pos_B);  // partial h / partial delta_gyro_bias

      auto h_cov = H * P_ * H.transpose() + vel_B_noise_cov;
      auto K = P_ * H.transpose() * (h_cov).inverse();
      auto delta_x = K * (-h);
      P_ = (Eigen::MatrixXd::Identity(12, 12) - K * H) * P_;

      // update nominal system
      vel_I_ += delta_x.block(0, 0, 3, 1);
      Eigen::Quaterniond delta_att;
      delta_att.vec() = delta_x.block(3, 0, 3, 1) * 0.5;
      delta_att.w() = 1;
      delta_att.normalize();
      att_ = att_ * delta_att;
      acc_bias_ += delta_x.block(6, 0, 3, 1);
      gyro_bias_ += delta_x.block(9, 0, 3, 1);
      std::cout << "zero contact vel update: " << std::endl;
      // std::cout << " att: " << att_.coeffs().transpose() << std::endl;
      // std::cout << " delta_att: " << delta_att.coeffs() << std::endl;
      // std::cout << " vel_I: " << vel_I_.transpose() << std::endl;
      std::cout << " delta x " << delta_x.transpose() << std::endl;
      std::cout << " h " << h.transpose() << std::endl;
      std::cout << P_ << std::endl;
      h_out = h;
      h_cov_out = h_cov;
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




class KinematicOdometry : public rclcpp::Node
{
public:
  KinematicOdometry()
  : Node("kinematic_odometry")
  {
    this->declare_parameter<std::vector<std::string>>("contact_joint_names");
    contact_joint_names_ = this->get_parameter("contact_joint_names").as_string_array();
    contact_states_.resize(contact_joint_names_.size());

    base_link_frame_id_ = this->declare_parameter<std::string>("base_link_frame_id", "base_link");
    odom_frame_id_ = this->declare_parameter<std::string>("odom_frame_id", "odom");

    // load noise parameters
    ekf_.vel_std_0_ = this->declare_parameter<double>("vel_std_0", 0.1);
    ekf_.att_std_0_ = this->declare_parameter<double>("att_std_0", 0.1);
    ekf_.acc_bias_std_0_ = this->declare_parameter<double>("acc_bias_std_0", 0.1);
    ekf_.gyro_bias_std_0_ = this->declare_parameter<double>("gyro_bias_std_0", 0.01);
    ekf_.acc_noise_density_ = this->declare_parameter<double>("acc_noise_density", 0.01);
    ekf_.gyro_noise_density_ = this->declare_parameter<double>("gyro_noise_density", 0.01);
    ekf_.acc_bias_random_walk_ = this->declare_parameter<double>("acc_bias_random_walk", 0.01);
    ekf_.gyro_bias_random_walk_ = this->declare_parameter<double>("gyro_bias_random_walk", 0.01);

    contact_timeout_ = this->declare_parameter<double>("contact_timeout", 0.03);
    joint_state_timeout_ = this->declare_parameter<double>("joint_state_timeout", 0.03);
    zero_velocity_timeout_ = this->declare_parameter<double>("zero_velocity_timeout", 0.2);
    zero_vel_noise_cov_ = Eigen::Matrix3d::Identity() * this->declare_parameter<double>("zero_vel_noise_cov", 1.0);
    contact_vel_covariance_ = Eigen::Matrix3d::Identity() * this->declare_parameter<double>("contact_vel_covariance", 0.1);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("~/odom", 10);
    ekf_innovations_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/ekf_innovations", 10);

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "~/imu", 10, std::bind(&KinematicOdometry::imu_cb, this, _1));

    robot_desc_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/robot_description", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(), std::bind(&KinematicOdometry::robot_desc_cb, this, _1));

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "~/joint_states", 10, std::bind(&KinematicOdometry::joint_state_cb, this, _1));

    for (size_t i = 0; i < contact_joint_names_.size(); i++)
    {
      auto name = contact_joint_names_[i];
      auto contact_sub = this->create_subscription<biped_bringup::msg::StampedBool>(
        "~/" + name + "_contact", 10, [this, i](const biped_bringup::msg::StampedBool::SharedPtr msg) {contact_cb(msg, i);});
      contact_subs_.push_back(contact_sub);
    }
  }

private:

  void robot_desc_cb(const std_msgs::msg::String::SharedPtr msg)
  {
    RCLCPP_INFO_ONCE(this->get_logger(), "received robot description");
    RCLCPP_WARN_SKIPFIRST(this->get_logger(), "robot description was updated");
    std::string urdf_xml_string = msg->data;
    pinocchio::urdf::buildModelFromXML(urdf_xml_string, pinocchio::JointModelFreeFlyer(), model_); // TODO is there a fixed option?
    RCLCPP_DEBUG_STREAM(this->get_logger(), "model nq:" << model_.nq);
    RCLCPP_DEBUG_STREAM(this->get_logger(), "model nv:" << model_.nv);
    RCLCPP_DEBUG_STREAM(this->get_logger(), "model njoints:" << model_.njoints);
    for (int i = 0; i < model_.njoints; i++)
    {
      RCLCPP_DEBUG_STREAM(this->get_logger(), "  joint " << i << " name:" << model_.names[i]);
    }

    if (!model_.existFrame(base_link_frame_id_))
    {
      RCLCPP_ERROR(this->get_logger(), "%s is not in the model", base_link_frame_id_.c_str());
      throw std::runtime_error("base_link_frame_id is not in the model");
    }
    for (size_t i = 0; i < contact_joint_names_.size(); i++)
    {
      if (!model_.existFrame(contact_joint_names_[i]))
      {
        RCLCPP_ERROR(this->get_logger(), "contact joint %s does not exist in the model", contact_joint_names_[i].c_str());
        throw std::runtime_error("contact joint does not exist in the model");
      }
    }
    model_loaded_ = true;
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

    // if no contact, perform zero velocity updates
    if ((time_ - last_contact_velocity_update_).seconds() > zero_velocity_timeout_)
    {
      if (!no_contact_zero_vel_update_active_) {
        ekf_.reset(q_IMU_to_BL_);
      }
      no_contact_zero_vel_update_active_ = true;
      // TODO we could put the base link position here
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 100 /* [ms] */, "No contact, performing zero velocity update");
      Eigen::Vector3d _h;
      Eigen::Matrix3d _h_cov;
      ekf_.zero_contact_vel_measurement_update(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), zero_vel_noise_cov_, _h, _h_cov);
    } else {
      no_contact_zero_vel_update_active_ = false;
    }

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
  }


  void joint_state_cb(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (time_ == rclcpp::Time(0, 0, RCL_ROS_TIME))
    {
      return;  // estimator not initialized yet
    }
    if (fabs((rclcpp::Time(msg->header.stamp) - time_).seconds()) > joint_state_timeout_)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100 /* [ms] */, "Joint state message time too far from imu time");
      return;
    }
    if (model_loaded_ == false) {
      RCLCPP_ERROR_SKIPFIRST_THROTTLE(this->get_logger(), *this->get_clock(), 1000 /* [ms] */, "No robot model loaded");
      return;
    }
    if (msg->position.size() != msg->name.size()) {
      RCLCPP_ERROR(this->get_logger(), "Invalid joint state: number of positions (%zu) does not match number of joints (%zu)", msg->position.size(), msg->name.size());
      return;
    }
    if (msg->velocity.size() != msg->name.size()) {
      RCLCPP_ERROR(this->get_logger(), "Invalid joint state: number of vels (%zu) does not match number of joints (%zu)", msg->velocity.size(), msg->name.size());
      return;
    }
    
    // initialize q for base_link
    assert(model_.existFrame(base_link_frame_id_));
    int base_link_joint_id = model_.frames[model_.getFrameId(base_link_frame_id_)].parent;
    const auto &base_link_joint = model_.joints[base_link_joint_id];
    assert(base_link_joint.nq() == 7);
    Eigen::VectorXd q(model_.nq);
    Eigen::VectorXd qvel(model_.nv);
    qvel.setZero();

    q[base_link_joint.idx_q()] = 0;
    q[base_link_joint.idx_q() + 1] = 0;
    q[base_link_joint.idx_q() + 2] = 0;
    q[base_link_joint.idx_q() + 3] = 0;
    q[base_link_joint.idx_q() + 4] = 0;
    q[base_link_joint.idx_q() + 5] = 0;
    q[base_link_joint.idx_q() + 6] = 1;

    int count_assignments = base_link_joint.nq();
    for (size_t i = 0; i < msg->name.size(); i++) {
        if (model_.existJointName(msg->name[i])) {
            int joint_id = model_.getJointId(msg->name[i]);
            const auto &joint = model_.joints[joint_id];
            // std::cout << msg->name[i] << " joint.idx_q() " << joint.idx_q() << " joint.idx_v()" << joint.idx_v() << std::endl;
            q[joint.idx_q()] = msg->position[i];
            qvel[joint.idx_v()] = msg->velocity[i];
            count_assignments++;
        }
    }
    if (count_assignments != model_.nq) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 100 /* [ms] */, "Joint state message does not contain all joints. Only %d of %d joints are assigned", count_assignments, model_.nq);
      return;
    }
    pinocchio::Data data(model_);
    pinocchio::computeJointJacobians(model_, data, q); // also computes forward kinematics
    pinocchio::updateFramePlacements(model_, data);

    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = base_link_frame_id_;
    marker.header.stamp = msg->header.stamp;
    marker.header.frame_id = imu_frame_id_;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.color.r = 1;
    marker.color.g = 0;
    marker.color.b = 0;
    marker.color.a = 1;
    marker.ns = "contact_vel";
    for (size_t i = 0; i < contact_states_.size(); i++) {
      if (contact_states_[i].data == true) {  // if in contact
        if ((time_ - rclcpp::Time(contact_states_[i].header.stamp)).seconds() > contact_timeout_) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100 /* [ms] */, "Contact state message time too old");
          continue;
        }
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 100 /* [ms] */, "performing contact update for %s", contact_joint_names_[i].c_str());
        Eigen::MatrixXd J(6, model_.nv); // contact frame jacobian in base_link (aligned with WORLD)
        J.setZero();
        auto frame_id = model_.getFrameId(contact_joint_names_[i]);
        pinocchio::getFrameJacobian(model_, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        Eigen::Vector3d p_contact_BL = data.oMf[frame_id].translation();
        Eigen::Vector3d v_contact_BL = J.block(0, 0, 3, model_.nv) * qvel; // velocity wrt base_link in base_link
        // transform to IMU frame
        auto R_BL_to_IMU = q_IMU_to_BL_.conjugate().toRotationMatrix();
        Eigen::Vector3d p_contact_IMU = R_BL_to_IMU * (p_contact_BL - p_IMU_in_BL_);
        Eigen::Vector3d v_contact_IMU = R_BL_to_IMU * v_contact_BL;
        Eigen::Vector3d _h;
        Eigen::Matrix3d _h_cov;
        ekf_.zero_contact_vel_measurement_update(p_contact_IMU, v_contact_IMU, contact_vel_covariance_, _h, _h_cov);
        last_contact_velocity_update_ = rclcpp::Time(msg->header.stamp);

        // markers
        Eigen::Vector3d posIMU_I, velIMU_I, omegaIMU_IMU;
        Eigen::Quaterniond att_IMU_to_I;
        ekf_.get_state(posIMU_I, velIMU_I, att_IMU_to_I, omegaIMU_IMU);
        // h marker
        marker.id = i;
        marker.pose.position.x = p_contact_IMU(0);
        marker.pose.position.y = p_contact_IMU(1);
        marker.pose.position.z = p_contact_IMU(2);
        Eigen::Vector3d h_IMU = att_IMU_to_I.conjugate() * _h;
        double h_norm = h_IMU.norm();
        std::cout << "h_norm " << h_norm << std::endl;
        std::cout << "h_IMU " << h_IMU.transpose() << std::endl;
        Eigen::Quaterniond h_dir = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), h_IMU);
        marker.pose.orientation.w = h_dir.w();
        marker.pose.orientation.x = h_dir.x();
        marker.pose.orientation.y = h_dir.y();
        marker.pose.orientation.z = h_dir.z();
        marker.scale.x = h_norm;
        marker.scale.y = 0.01 * h_norm;
        marker.scale.z = 0.01 * h_norm;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker_array.markers.push_back(marker);
      } else {
        marker.id = i;
        marker.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.push_back(marker);
      }
    }
    ekf_innovations_marker_pub_->publish(marker_array);
  }

  void contact_cb(const biped_bringup::msg::StampedBool::SharedPtr msg, int joint_idx)
  {
    contact_states_[joint_idx] = *msg;
  }

  rclcpp::Time time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  IMUPoseEKF ekf_;

  std::string imu_frame_id_;
  std::string base_link_frame_id_;
  std::string odom_frame_id_;
  Eigen::Quaterniond q_IMU_to_BL_;
  Eigen::Vector3d p_IMU_in_BL_;

  // noise parameters
  Eigen::Matrix3d zero_vel_noise_cov_;
  Eigen::Matrix3d contact_vel_covariance_;

  std::vector<std::string> contact_joint_names_;
  std::vector<biped_bringup::msg::StampedBool> contact_states_;

  double contact_timeout_;
  double joint_state_timeout_;
  double zero_velocity_timeout_;
  bool no_contact_zero_vel_update_active_ = false;

  pinocchio::Model model_;
  bool model_loaded_ = false;

  rclcpp::Time last_contact_velocity_update_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  std::vector<rclcpp::Subscription<biped_bringup::msg::StampedBool>::SharedPtr> contact_subs_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_desc_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ekf_innovations_marker_pub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KinematicOdometry>());
  rclcpp::shutdown();
  return 0;
}
