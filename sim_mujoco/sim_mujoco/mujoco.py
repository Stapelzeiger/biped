import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory, JointTrajectory
from geometry_msgs.msg import TransformStamped, Vector3Stamped, PoseStamped, PoseWithCovarianceStamped, TwistStamped

from rosgraph_msgs.msg import Clock
from biped_bringup.msg import StampedBool
from std_msgs.msg import Bool, Float64, Empty, Float32, String
from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R
import mujoco as mj
import mujoco_viewer
import numpy as np
import sys
import json
import time
from sim_mujoco.submodules.pid import pid as pid_ctrl

from threading import Lock
import math 

import csv
import os
sys.path.insert(0, '/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/sim_mujoco/sim_mujoco/')
from bc_controller import PolicyBC

def setup_pid(control_rate, kp, ki, kd):
    pid = pid_ctrl()
    pid.pid_set_frequency(control_rate)
    pid.pid_set_gains(kp, ki, kd)
    return pid

class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.declare_parameter("mujoco_xml_path", rclpy.parameter.Parameter.Type.STRING)
        self.declare_parameter("sim_time_sec", rclpy.parameter.Parameter.Type.DOUBLE)
        self.declare_parameter("visualization_rate", rclpy.parameter.Parameter.Type.DOUBLE)
        self.declare_parameter("visualize_mujoco", rclpy.parameter.Parameter.Type.BOOL)
        self.visualize_mujoco = self.get_parameter("visualize_mujoco").get_parameter_value().bool_value
        mujoco_xml_path = self.get_parameter("mujoco_xml_path").get_parameter_value().string_value
        self.sim_time_sec = self.get_parameter("sim_time_sec").get_parameter_value().double_value
        self.visualization_rate = self.get_parameter("visualization_rate").get_parameter_value().double_value
        self.initialization_done = False
        self.goal_pos = [0.0, 0.0]
        self.contact_states = {'R_FOOT': False,
                               'L_FOOT': False}

        self.model = mj.MjModel.from_xml_path(mujoco_xml_path)
        mj.mj_printModel(self.model, 'robot_information.txt')
        self.data = mj.MjData(self.model)
        self.lock = Lock()

        self.time = time.time()
        self.model.opt.timestep = self.sim_time_sec
        self.dt = self.model.opt.timestep
        self.R_b_to_I = None
        self.v_b = None
        self.swing_foot_BF_pos = None
        self.stance_foot_BF_pos = None
        self.dcm_desired_BF = None
        self.T_since_contact_right = 0.0
        self.T_since_contact_left = 0.0
        self.T_since_no_contact_right = 0.0
        self.T_since_no_contact_left = 0.0

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

        if self.visualize_mujoco == True:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -25
            self.viewer.render()

        self.name_joints = self.get_joint_names()

        self.q_joints = {}
        for i in self.name_joints:
            self.q_joints[i] = {
                'actual_pos': 0.0,
                'actual_vel': 0.0,
                'desired_pos': 0.0,
                'desired_vel': 0.0,
                'feedforward_torque': 0.0
            }

        self.name_actuators = []
        for i in range(0, self.model.nu):  # skip root
            self.name_actuators.append(mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i))

        self.q_actuator_addr = {}
        for name in self.name_actuators:
            self.q_actuator_addr[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)

        self.q_pos_addr_joints = {}
        for name in self.name_joints:
            self.q_pos_addr_joints[name] = self.model.jnt_qposadr[mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, name)]
        
        self.counter = 0

        self.init([0.0, 0.0, 0.0])

        self.clock_pub = self.create_publisher(Clock, '/clock', 10)

        self.odometry_base_pub = self.create_publisher(Odometry, '~/odometry', 10)
        self.contact_right_pub = self.create_publisher(StampedBool, '~/contact_foot_right', 10)
        self.contact_left_pub = self.create_publisher(StampedBool, '~/contact_foot_left', 10)

        self.joint_states_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '~/imu', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.joint_traj_sub = self.create_subscription(JointTrajectory, 'joint_trajectory', self.joint_traj_cb, 10)
        self.joint_traj_msg = None

        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.init_cb, 10)
        self.reset_sub = self.create_subscription(Empty, '~/reset', self.reset_cb, 10)

        self.swing_foot_BF_sub = self.create_subscription(Vector3Stamped, "~/swing_foot_BF", self.swing_foot_BF_cb, 10)
        self.stance_foot_BF_sub = self.create_subscription(Vector3Stamped, "~/stance_foot_BF", self.stance_foot_BF_cb, 10)
        self.dcm_desired_BF_sub = self.create_subscription(TwistStamped, "~/dcm_desired_BF", self.dcm_desired_BF_cb, 10)

        self.paused = True
        self.step_sim_sub = self.create_subscription(Float64, "~/step", self.step_cb, 1)
        self.pause_sim_sub = self.create_subscription(Bool, "~/pause", self.pause_cb, 1)

        self.folder_name = 'src/biped/sim_mujoco/sim_mujoco/data'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        self.file = open(f'{self.folder_name}/dataset.csv', 'w', newline='')
        self.writer = csv.writer(self.file)
        self.write_controller_dataset_header()

        self.policy_NN = None
        self.policy_input_size = len(self.name_joints) * 2 + 6 + 2 + 5 * 2
        action_size = len(self.name_joints) * 3 # q_des, qdot_des, tau_ff
        self.num_input_state = 3
        self.policy_bc_NN = PolicyBC(self.policy_input_size, action_size, self.num_input_state)
        self.list_policy_inputs = []
        self.timer = self.create_timer(self.dt, self.timer_cb)

    def swing_foot_BF_cb(self, msg):
        self.swing_foot_BF_pos = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def stance_foot_BF_cb(self, msg):
        self.stance_foot_BF_pos = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def dcm_desired_BF_cb(self, msg):
        self.dcm_desired_BF = np.array([msg.twist.linear.x, msg.twist.linear.y])

    def reset_cb(self, msg):
        with self.lock:
            self.init([0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

    def init_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self.init([p.x, p.y, p.z], q=[q.w, q.x, q.y, q.z])

    def init(self, p, q=[1.0, 0.0, 0.0, 0.0]):
        self.model.eq_data[0][0] = p[0]
        self.model.eq_data[0][1] = p[1]
        self.model.eq_data[0][2] = 1.5 # one meter above gnd

        self.data.qpos = [0.0] * self.model.nq
        self.data.qpos[3] = q[0]
        self.data.qpos[4] = q[1]
        self.data.qpos[5] = q[2]
        self.data.qpos[6] = q[3]

        self.data.qvel = [0.0]* self.model.nv

        self.data.qpos[0] = p[0]
        self.data.qpos[1] = p[1]
        self.data.qpos[2] = self.model.eq_data[0][2]

        self.model.eq_active[0] = 1
        mj.mj_step(self.model, self.data)
        self.initialization_done = False
        self.get_logger().info("initialize")

    def timer_cb(self):
        with self.lock:
            if not self.paused:
                self.step()

    def step_cb(self, msg):
        with self.lock:
            if not self.paused:
                return

            t = msg.data
            while t > 0:
                t -= self.dt
                self.step()

    def pause_cb(self, msg):
        with self.lock:
            self.paused = msg.data

    def step(self):
        if not self.initialization_done:
            self.model.eq_data[0][2] -= 0.5 * self.dt

        self.read_contact_states()
        if self.contact_states['R_FOOT'] or self.contact_states['L_FOOT']:
            if not self.initialization_done:
                self.get_logger().info("init done")
                self.initialization_done = True
                self.data.qvel = [0.0]* self.model.nv
                self.model.eq_active[0] = 0 # let go of the robot

        if self.visualize_mujoco is True:
            vis_update_downsampling = int(round(1.0/self.visualization_rate/self.sim_time_sec/10))
            if self.counter % vis_update_downsampling == 0:
                self.viewer.render()

        if self.initialization_done and self.dcm_desired_BF is not None:
            policy_input_one_instance = self.create_policy_input()

            while len(self.list_policy_inputs) < self.policy_input_size * self.num_input_state:
                self.list_policy_inputs.extend(policy_input_one_instance)

            self.list_policy_inputs.extend(policy_input_one_instance) # add new one
            for i in range(self.policy_input_size):
                self.list_policy_inputs.pop(0) # pop the old state
            policy_input = np.array(self.list_policy_inputs)
            self.policy_NN = self.policy_bc_NN(policy_input)

        self.run_joint_controllers(self.policy_NN)

        mj.mj_step(self.model, self.data)
        self.time += self.dt
        self.counter += 1

        clock_msg = Clock()
        clock_msg.clock.sec = int(self.time)
        clock_msg.clock.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        self.clock_pub.publish(clock_msg)

        msg_odom = Odometry()
        msg_odom.header.stamp.sec = int(self.time)
        msg_odom.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_odom.header.frame_id = 'odom'
        msg_odom.child_frame_id = 'base_link'
        msg_odom.pose.pose.position.x = self.data.qpos[0]
        msg_odom.pose.pose.position.y = self.data.qpos[1]
        msg_odom.pose.pose.position.z = self.data.qpos[2]
        msg_odom.pose.pose.orientation.w = self.data.qpos[3]
        msg_odom.pose.pose.orientation.x = self.data.qpos[4]
        msg_odom.pose.pose.orientation.y = self.data.qpos[5]
        msg_odom.pose.pose.orientation.z = self.data.qpos[6]
        q = msg_odom.pose.pose.orientation

        self.R_b_to_I = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self.v_b = self.R_b_to_I.T @ self.data.qvel[0:3] # linear vel is in inertial frame
        msg_odom.twist.twist.linear.x = self.v_b[0]
        msg_odom.twist.twist.linear.y = self.v_b[1]
        msg_odom.twist.twist.linear.z = self.v_b[2]
        msg_odom.twist.twist.angular.x = self.data.qvel[3] # angular vel is in body frame
        msg_odom.twist.twist.angular.y = self.data.qvel[4]
        msg_odom.twist.twist.angular.z = self.data.qvel[5]
        self.odometry_base_pub.publish(msg_odom)

        t = TransformStamped()
        t.header = msg_odom.header
        t.child_frame_id = msg_odom.child_frame_id
        t.transform.translation.x = msg_odom.pose.pose.position.x
        t.transform.translation.y = msg_odom.pose.pose.position.y
        t.transform.translation.z = msg_odom.pose.pose.position.z
        t.transform.rotation = msg_odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

        self.read_contact_states()
        msg_contact_right = StampedBool()
        msg_contact_left = StampedBool()
        msg_contact_right.header.stamp.sec = int(self.time)
        msg_contact_right.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_contact_right.data = self.contact_states['R_FOOT']
        msg_contact_left.header.stamp.sec = int(self.time)
        msg_contact_left.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_contact_left.data = self.contact_states['L_FOOT']
        self.contact_right_pub.publish(msg_contact_right)
        self.contact_left_pub.publish(msg_contact_left)

        msg_joint_states = JointState()
        msg_joint_states.header.stamp.sec = int(self.time)
        msg_joint_states.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            value['actual_acc'] = self.data.qacc[self.model.jnt_dofadr[id_joint_mj]]
            msg_joint_states.name.append(key)
            msg_joint_states.position.append(value['actual_pos'])
            msg_joint_states.velocity.append(value['actual_vel'])
            msg_joint_states.effort.append(value['actual_acc'])
        self.joint_states_pub.publish(msg_joint_states)

        self.ankle_foot_spring('L_ANKLE')
        self.ankle_foot_spring('R_ANKLE')

        gyro_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "gyro")
        accel_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "accelerometer")
        accel = self.data.sensordata[self.model.sensor_adr[accel_id]:self.model.sensor_adr[accel_id] + 3]
        self.accel_bias += np.random.normal(0, self.accel_bias_noise_std, 3)
        # accel += self.accel_bias + np.random.normal(0, self.accel_noise_std, 3)
        gyro = self.data.sensordata[self.model.sensor_adr[gyro_id]:self.model.sensor_adr[gyro_id] + 3]
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

        if self.initialization_done and self.dcm_desired_BF is not None:
            policy_input_for_recording = self.create_policy_input()
            policy_output_for_recording = self.create_policy_output()
            self.write_controller_dataset_entry(policy_input_for_recording, policy_output_for_recording)

    def run_joint_controllers(self, policy_NN=None):
        if self.joint_traj_msg is None:
            return

        print('running joint controllers')
        print('policy NN:', policy_NN)
        print(self.name_joints)
        if policy_NN is None:
            # running the expert policy
            for key, value in self.q_joints.items():
                id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
                value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
                value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
                value['actual_acc'] = self.data.qacc[self.model.jnt_dofadr[id_joint_mj]]
                if key in self.joint_traj_msg.joint_names:
                    id_joint_msg = self.joint_traj_msg.joint_names.index(key)
                    value['desired_pos'] = self.joint_traj_msg.points[0].positions[id_joint_msg]
                    if self.joint_traj_msg.points[0].velocities:
                        value['desired_vel'] = self.joint_traj_msg.points[0].velocities[id_joint_msg]
                    if self.joint_traj_msg.points[0].effort:
                        if math.isnan(self.joint_traj_msg.points[0].effort[id_joint_msg]) is False:
                            value['feedforward_torque'] = self.joint_traj_msg.points[0].effort[id_joint_msg]
                        else:
                            value['feedforward_torque'] = 0.0
        else:
            desired_pos = policy_NN[0: len(self.name_joints)]
            desired_vel = policy_NN[len(self.name_joints): 2 * len(self.name_joints)]
            tau_ff = policy_NN[2 * len(self.name_joints): 3 * len(self.name_joints)]
            # bad to duplicate code, will fix later
            cnt = 0
            for key, value in self.q_joints.items():
                id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
                print(key)
                value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
                value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
                value['actual_acc'] = self.data.qacc[self.model.jnt_dofadr[id_joint_mj]]
                if (key != 'L_ANKLE' or key != 'R_ANKLE'):
                    value['desired_pos'] = desired_pos[cnt]
                    value['desired_vel'] = desired_vel[cnt]
                    value['feedforward_torque'] = tau_ff[cnt]
                cnt += 1


        kp_moteus = 600.0
        Kp = (kp_moteus/(2*math.pi)) * np.ones(self.model.njnt - 1) # exclude root

        i = 0
        for key, value in self.q_joints.items():
            if key != 'L_ANKLE' and key != 'R_ANKLE':
                error = value['actual_pos'] - value['desired_pos']
                actuators_torque = - Kp[i]*error
                actuators_vel = value['desired_vel']
                feedforward_torque = value['feedforward_torque']
                self.data.ctrl[self.q_actuator_addr[str(key)]] = actuators_torque
                self.data.ctrl[self.q_actuator_addr[str(key) + "_VEL"]] = actuators_vel
                id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
                self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]] = feedforward_torque
            i = i + 1

    def write_controller_dataset_header(self):
        # Dataset: 
        #   joint states
        #   baselink vel BF
        #   goal vx_des_BF, vy_des BF
        #   right & left foot: 
        #           t since contact,
        #           t since no contact, 
        #           pos BF
        #   controller: tau_ff, q_j_des, q_j_vel_des

        header_for_joint_states = [name + '_pos' for name in self.name_joints]
        header_for_joint_states += [name + '_vel' for name in self.name_joints]
        header_for_baselink = ['vel_x_BF', 'vel_y_BF', 'vel_z_BF', 'normal_vec_x_BF', 'normal_vec_y_BF', 'normal_vec_z_BF', 'omega_x', 'omega_y', 'omega_z']
        header_for_goal = ['vx_des_BF', 'vy_des_BF']
        header_for_right_foot = ['right_foot_t_since_contact', 'right_foot_t_since_no_contact', 'right_foot_pos_x_BF', 'right_foot_pos_y_BF', 'right_foot_pos_z_BF']
        header_for_left_foot = ['left_foot_t_since_contact', 'left_foot_t_since_no_contact', 'left_foot_pos_x_BF', 'left_foot_pos_y_BF', 'left_foot_pos_z_BF']
        header_for_tau_ff = [name + '_tau_ff' for name in self.name_joints]
        header_for_q_j_des =  [name + '_q_des' for name in self.name_joints]
        header_for_q_j_vel_des = [name + '_q_vel des' for name in self.name_joints]
        header = ['time', *header_for_joint_states,
                          *header_for_baselink,
                          *header_for_goal, 
                          *header_for_right_foot,
                          *header_for_left_foot,
                          *header_for_tau_ff, *header_for_q_j_des, *header_for_q_j_vel_des]
        print(header)
        self.writer.writerow(header)

    def create_policy_input(self):
        data_for_joint_states = [self.q_joints[name]['actual_pos'] for name in self.name_joints]
        data_for_joint_states += [self.q_joints[name]['actual_vel'] for name in self.name_joints]
        normal_vector_I = np.array([0.0, 0.0, 1.0])
        normal_vector_BF = self.R_b_to_I.T@normal_vector_I
        data_for_baselink = [self.v_b[0], self.v_b[1], self.v_b[2],
                            normal_vector_BF[0], normal_vector_BF[1], normal_vector_BF[2],
                            self.data.qvel[3], self.data.qvel[4], self.data.qvel[5]]
        data_for_goal = [self.dcm_desired_BF[0], self.dcm_desired_BF[1]]

        if self.contact_states['R_FOOT'] == True:
            data_for_right_foot = [self.stance_foot_BF_pos[0], self.stance_foot_BF_pos[1], # z is removed, todo fix why it is always 0
                                self.T_since_contact_right, self.T_since_no_contact_right]
        else:
            data_for_right_foot = [self.swing_foot_BF_pos[0], self.swing_foot_BF_pos[1],  # z is removed, todo fix why it is always 0
                                self.T_since_contact_right, self.T_since_contact_right]

        if self.contact_states['L_FOOT'] == True:
            data_for_left_foot = [self.stance_foot_BF_pos[0], self.stance_foot_BF_pos[1], self.stance_foot_BF_pos[2],
                                self.T_since_contact_left, self.T_since_no_contact_left]
        else:
            data_for_left_foot = [self.swing_foot_BF_pos[0], self.swing_foot_BF_pos[1], self.swing_foot_BF_pos[2],
                                self.T_since_contact_left, self.T_since_no_contact_left]
            
        print('right foot:', data_for_right_foot)
        print('left foot:', data_for_left_foot)
        policy_input = [*data_for_joint_states,
                        *data_for_baselink,
                        *data_for_goal,
                        *data_for_right_foot,
                        *data_for_left_foot]
        return policy_input

    def create_policy_output(self):
        data_for_tau_ff = [self.q_joints[name]['feedforward_torque'] for name in self.name_joints]
        data_for_q_j_des = [self.q_joints[name]['desired_pos'] for name in self.name_joints]
        data_for_q_j_vel_des = [self.q_joints[name]['desired_vel'] for name in self.name_joints]
        policy_output = [*data_for_tau_ff, *data_for_q_j_des, *data_for_q_j_vel_des]
        return policy_output

    def write_controller_dataset_entry(self, policy_input, policy_output):
        if self.initialization_done and self.dcm_desired_BF is not None:
            row_entry = [self.time, *policy_input, *policy_output]
            self.writer.writerow(row_entry)

    def joint_traj_cb(self, msg):
        with self.lock:
            self.joint_traj_msg = msg

    def read_contact_states(self):
        self.contact_states['R_FOOT'] = False
        self.contact_states['L_FOOT'] = False

        geom1_list = []
        geom2_list = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            name_geom1 = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1)
            name_geom2 = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2)
            geom1_list.append(name_geom1)
            geom2_list.append(name_geom2)

        if self.data.ncon != 0:
            if 'L_FOOT' in geom2_list:
                first_entry_idx = geom2_list.index('L_FOOT')
                if geom1_list[first_entry_idx] == 'floor':
                    self.contact_states['L_FOOT'] = True

            if 'R_FOOT' in geom2_list:
                first_entry_idx = geom2_list.index('R_FOOT')
                if geom1_list[first_entry_idx] == 'floor':
                    self.contact_states['R_FOOT'] = True

        if self.contact_states['R_FOOT'] == True:
            self.T_since_contact_right = self.T_since_contact_right + self.dt
            self.T_since_no_contact_right = 0.0

        if self.contact_states['R_FOOT'] == False:
            self.T_since_contact_right = 0.0
            self.T_since_no_contact_right = self.T_since_no_contact_right + self.dt

        if self.contact_states['L_FOOT'] == True:
            self.T_since_contact_left = self.T_since_contact_left + self.dt
            self.T_since_no_contact_right = 0.0

        if self.contact_states['L_FOOT'] == False:
            self.T_since_contact_left = 0.0
            self.T_since_no_contact_left = self.T_since_no_contact_left + self.dt

    def get_joint_names(self):
        self.name_joints = []
        for i in range(1, self.model.njnt):  # skip root
            self.name_joints.append(mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_JOINT, i))
        return self.name_joints

    def ankle_foot_spring(self, foot_joint):
        'ankle modelled as a spring damped system'
        K = 0.005
        offset = 0.5
        pitch_error_foot = self.data.qpos[self.q_pos_addr_joints[foot_joint]] - offset
        pitch_torque_setpt = - K * pitch_error_foot

        idx_act = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_ACTUATOR, foot_joint)
        vel_actuator_name = str(foot_joint) + str('_VEL')
        idx_vel_act = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_ACTUATOR, vel_actuator_name)

        self.data.ctrl[idx_act] = pitch_torque_setpt
        self.data.ctrl[idx_vel_act] = 0.0

def main(args=None):
    rclpy.init(args=args)
    sim_node = MujocoNode()
    rclpy.spin(sim_node)
    sim_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
