import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseWithCovarianceStamped, TwistStamped

from rosgraph_msgs.msg import Clock
from biped_bringup.msg import StampedBool
from std_msgs.msg import Bool, Float64, Empty
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R

import numpy as np
import time
from threading import Lock
import sim_mujoco.submodules.mujoco_env_sim as mujoco_sim

class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.get_logger().info("Start Sim!")
        self.declare_parameter("mujoco_xml_path", rclpy.parameter.Parameter.Type.STRING)
        self.declare_parameter("sim_time_sec", rclpy.parameter.Parameter.Type.DOUBLE)
        self.declare_parameter("visualize_mujoco", rclpy.parameter.Parameter.Type.BOOL)
        self.declare_parameter("publish_tf", rclpy.parameter.Parameter.Type.BOOL)
        self.declare_parameter("use_RL_controller", rclpy.parameter.Parameter.Type.BOOL)
        self.visualize_mujoco = self.get_parameter("visualize_mujoco").get_parameter_value().bool_value
        mujoco_xml_path = self.get_parameter("mujoco_xml_path").get_parameter_value().string_value
        self.sim_time_sec = self.get_parameter("sim_time_sec").get_parameter_value().double_value
        self.publish_tf = self.get_parameter("publish_tf").get_parameter_value().bool_value
        self.use_RL_controller = self.get_parameter("use_RL_controller").get_parameter_value().bool_value
        self.initialization_done = False

        self.biped = mujoco_sim.Biped(mujoco_xml_path, self.sim_time_sec, self.visualize_mujoco, self.use_RL_controller)

        self.lock = Lock()

        self.time = time.time()
        self.dt = self.biped.get_sim_dt()

        self.accel_noise_density = 0.14 * 9.81/1000 # [m/s2 * sqrt(s)]
        self.accel_bias_random_walk = 0.0004 # [m/s2 / sqrt(s)]
        self.gyro_noise_density = 0.0035 / 180 * np.pi # [rad/s * sqrt(s)]
        self.gyro_bias_random_walk = 8.0e-06 # [rad/s / sqrt(s)]
        self.accel_noise_std = self.accel_noise_density / np.sqrt(self.dt)
        self.accel_bias_noise_std = self.accel_bias_random_walk * np.sqrt(self.dt)
        self.gyro_noise_std = self.gyro_noise_density / np.sqrt(self.dt)
        self.gyro_bias_noise_std = self.gyro_bias_random_walk * np.sqrt(self.dt)
        self.accel_bias = np.random.normal(0, self.accel_bias_random_walk * np.sqrt(100), 3)
        self.gyro_bias = np.random.normal(0, self.gyro_bias_random_walk * np.sqrt(100), 3)

        self.name_joints = self.biped.get_joint_names()
        self.q_joints = self.biped.get_q_joints_dict()

        self.name_actuators = self.biped.get_name_actuators()
        self.q_actuator_addr = self.biped.get_q_actuator_addr()
        self.q_pos_addr_joints = self.biped.get_q_pos_addr_joints()

        self.counter = 0
        self.init(p=[0.0, 0.0, 0.0])

        # Publishers.
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.odometry_base_pub = self.create_publisher(Odometry, '~/odometry', 10)
        self.contact_right_pub = self.create_publisher(StampedBool, '~/contact_foot_right', 10)
        self.contact_left_pub = self.create_publisher(StampedBool, '~/contact_foot_left', 10)

        self.qfrc_actuators_pub = self.create_publisher(JointState, '~/qfrc_actuators', 10)
        self.tau_actuators_pub = self.create_publisher(JointState, '~/tau_actuators', 10)
        self.qfrc_passive_pub = self.create_publisher(JointState, '~/qfrc_passive', 10)

        self.stop_pub = self.create_publisher(Bool, '~/stop', 10)

        self.joint_states_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '~/imu', 10)
        self.fake_vicon_pub = self.create_publisher(PoseStamped, '~/fake_vicon', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers.
        self.joint_traj_sub = self.create_subscription(JointTrajectory, 'joint_trajectory', self.joint_traj_cb, 10)
        self.joint_traj_msg = None
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.init_cb, 10)
        self.reset_sub = self.create_subscription(Empty, '~/reset', self.reset_cb, 10)

        self.paused = False
        self.step_sim_sub = self.create_subscription(Float64, "~/step", self.step_cb, 1)
        self.pause_sim_sub = self.create_subscription(Bool, "~/pause", self.pause_cb, 1)

        self.cmd = None
        self.cmd_vel_sub = self.create_subscription(TwistStamped, "/vel_cmd", self.cmd_vel_cb, 1)

        self.timer = self.create_timer(self.dt*2, self.timer_cb, clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME))

    def cmd_vel_cb(self, msg: TwistStamped):
        self.cmd = np.zeros(3)
        self.cmd[0] = msg.twist.linear.x
        self.cmd[1] = msg.twist.linear.y
        self.cmd[2] = msg.twist.angular.z

    def reset_cb(self, msg: Empty):
        with self.lock:
            self.get_logger().info("Reset")
            self.init(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

    def init_cb(self, msg: PoseWithCovarianceStamped):
        with self.lock:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self.init([p.x, p.y, p.z], q=[q.w, q.x, q.y, q.z])

    def init(self, p: list, q: list =[1.0, 0.0, 0.0, 0.0]):
        self.biped.init(p, q)
        self.get_logger().info("initialize")
        self.initialization_done = False
        self.initialization_timeout = 0.2

    def timer_cb(self):
        with self.lock:
            if not self.paused:
                self.step()

    def step_cb(self, msg: Float64):
        with self.lock:
            if not self.paused:
                return

            t = msg.data
            while t > 0:
                t -= self.dt
                self.step()

    def pause_cb(self, msg: Bool):
        with self.lock:
            self.paused = msg.data

    def stop_viz(self):
        self.biped.viewer.close()

    def step(self):
        if not self.initialization_done:
            msg_stop_controller = Bool()
            msg_stop_controller.data = (self.initialization_timeout > 0)
            self.stop_pub.publish(msg_stop_controller)
            self.biped.lower_robot(step=self.dt)
            self.initialization_timeout -= self.dt

        contact_states = self.biped.read_contact_states()
        if contact_states['R_FOOT'] or contact_states['L_FOOT']:
            if not self.initialization_done:
                self.get_logger().info("init done")
                self.initialization_done = True
                self.biped.zero_the_velocities()
                self.biped.let_go_of_the_robot()

        # Step the simulation.
        for _ in range(2):
            is_valid_traj_msg = False if self.joint_traj_msg is None else True
            # # Build a joint_traj_dict.
            if is_valid_traj_msg is True:
                joint_traj_dict = {}
                for name in self.joint_traj_msg.joint_names:
                    joint_traj_dict[name] = {
                        'pos': self.joint_traj_msg.points[0].positions[self.joint_traj_msg.joint_names.index(name)],
                        'vel': self.joint_traj_msg.points[0].velocities[self.joint_traj_msg.joint_names.index(name)],
                        'effort': self.joint_traj_msg.points[0].effort[self.joint_traj_msg.joint_names.index(name)]
                    }
            else:

                joint_traj_dict = None

            qpos, qvel = self.biped.step(joint_traj_dict)
            self.time += self.dt
            self.counter += 1

        self.q_joints = self.biped.get_q_joints_dict()

        # ROS publishers.
        clock_msg = Clock()
        clock_msg.clock.sec = int(self.time)
        clock_msg.clock.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        self.clock_pub.publish(clock_msg)

        msg_odom = Odometry()
        msg_odom.header.stamp.sec = int(self.time)
        msg_odom.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_odom.header.frame_id = 'odom'
        msg_odom.child_frame_id = 'base_link'
        msg_odom.pose.pose.position.x = qpos[0]
        msg_odom.pose.pose.position.y = qpos[1]
        msg_odom.pose.pose.position.z = qpos[2]
        msg_odom.pose.pose.orientation.w = qpos[3]
        msg_odom.pose.pose.orientation.x = qpos[4]
        msg_odom.pose.pose.orientation.y = qpos[5]
        msg_odom.pose.pose.orientation.z = qpos[6]
        q = msg_odom.pose.pose.orientation

        R_b_to_I = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        v_b = R_b_to_I.T @ qvel[0:3] # linear vel is in inertial frame
        msg_odom.twist.twist.linear.x = v_b[0]
        msg_odom.twist.twist.linear.y = v_b[1]
        msg_odom.twist.twist.linear.z = v_b[2]
        msg_odom.twist.twist.angular.x = qvel[3] # angular vel is in body frame
        msg_odom.twist.twist.angular.y = qvel[4]
        msg_odom.twist.twist.angular.z = qvel[5]
        self.odometry_base_pub.publish(msg_odom)

        if self.publish_tf:
            t = TransformStamped()
            t.header = msg_odom.header
            t.child_frame_id = msg_odom.child_frame_id
            t.transform.translation.x = msg_odom.pose.pose.position.x
            t.transform.translation.y = msg_odom.pose.pose.position.y
            t.transform.translation.z = msg_odom.pose.pose.position.z
            t.transform.rotation = msg_odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)

        contact_states = self.biped.read_contact_states()
        msg_contact_right = StampedBool()
        msg_contact_left = StampedBool()
        msg_contact_right.header.stamp.sec = int(self.time)
        msg_contact_right.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_contact_right.data = contact_states['R_FOOT']
        msg_contact_left.header.stamp.sec = int(self.time)
        msg_contact_left.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_contact_left.data = contact_states['L_FOOT']
        self.contact_right_pub.publish(msg_contact_right)
        self.contact_left_pub.publish(msg_contact_left)

        msg_joint_states = JointState()
        msg_joint_states.header.stamp.sec = int(self.time)
        msg_joint_states.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        for key, value in self.q_joints.items():
            msg_joint_states.name.append(key)
            msg_joint_states.position.append(value['actual_pos'])
            msg_joint_states.velocity.append(value['actual_vel'])
            msg_joint_states.effort.append(value['actual_effort'])
        self.joint_states_pub.publish(msg_joint_states)

        # qfrc message
        msg_qfrc_actuators = JointState()
        msg_qfrc_actuators.header.stamp.sec = int(self.time)
        msg_qfrc_actuators.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        for key, value in self.q_joints.items():
            msg_qfrc_actuators.name.append(key)
            msg_qfrc_actuators.effort.append(value['qfrc_actuator'])
        self.qfrc_actuators_pub.publish(msg_qfrc_actuators)

        # tau
        msg_tau_actuators = JointState()
        msg_tau_actuators.header.stamp.sec = int(self.time)
        msg_tau_actuators.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        for key, value in self.q_joints.items():
            msg_tau_actuators.name.append(key)
            msg_tau_actuators.effort.append(value['total_tau'])
        self.tau_actuators_pub.publish(msg_tau_actuators)

        # qfrc passive
        msg_qfrc_passive = JointState()
        msg_qfrc_passive.header.stamp.sec = int(self.time)
        msg_qfrc_passive.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        for key, value in self.q_joints.items():
            msg_qfrc_passive.name.append(key)
            msg_qfrc_passive.effort.append(value['qfrc_passive'])
        self.qfrc_passive_pub.publish(msg_qfrc_passive)

        gyro = self.biped.get_sensor_data(name='gyro')
        accel = self.biped.get_sensor_data(name='accelerometer')
        self.accel_bias += np.random.normal(0, self.accel_bias_noise_std, 3)
        # accel += self.accel_bias + np.random.normal(0, self.accel_noise_std, 3)
        self.gyro_bias += np.random.normal(0, self.gyro_bias_noise_std, 3)
        # gyro += self.gyro_bias + np.random.normal(0, self.gyro_noise_std, 3)
        if not self.initialization_done:
            accel = [0.0, 0.0, -9.81]
            gyro = np.zeros(3)
        msg_imu = Imu()
        msg_imu.header.stamp.sec = int(self.time)
        msg_imu.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_imu.header.frame_id = 'imu'
        msg_imu.linear_acceleration.x = accel[0]
        msg_imu.linear_acceleration.y = accel[1]
        msg_imu.linear_acceleration.z = accel[2]
        msg_imu.angular_velocity.x = gyro[0]
        msg_imu.angular_velocity.y = gyro[1]
        msg_imu.angular_velocity.z = gyro[2]
        msg_imu.orientation_covariance[0] = -1 # no orientation
        self.imu_pub.publish(msg_imu)

        msg_fake_vicon = PoseStamped()
        msg_fake_vicon.header.stamp.sec = int(self.time)
        msg_fake_vicon.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_fake_vicon.header.frame_id = 'world'
        msg_fake_vicon.pose.position.x = qpos[0]
        msg_fake_vicon.pose.position.y = qpos[1]
        msg_fake_vicon.pose.position.z = qpos[2]
        msg_fake_vicon.pose.orientation.w = qpos[3]
        msg_fake_vicon.pose.orientation.x = qpos[4]
        msg_fake_vicon.pose.orientation.y = qpos[5]
        msg_fake_vicon.pose.orientation.z = qpos[6]
        self.fake_vicon_pub.publish(msg_fake_vicon)

    def joint_traj_cb(self, msg: JointTrajectory):
        with self.lock:
            self.joint_traj_msg = msg

def main(args=None):
    rclpy.init(args=args)
    sim_node = MujocoNode()
    rclpy.spin(sim_node)
    sim_node.stop_viz()
    sim_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
