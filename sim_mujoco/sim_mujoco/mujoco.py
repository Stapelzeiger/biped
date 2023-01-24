import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory, JointTrajectory
from geometry_msgs.msg import TransformStamped, Vector3, PoseStamped, PoseWithCovarianceStamped

from rosgraph_msgs.msg import Clock
from biped_bringup.msg import StampedBool
from std_msgs.msg import Bool, Float64, Empty
from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R

import mujoco as mj
import mujoco_viewer
import numpy as np
import sys
import json
from sim_mujoco.submodules.pid import pid as pid_ctrl

from threading import Lock

def setup_pid(control_rate, kp, ki, kd):
    pid = pid_ctrl()
    pid.pid_set_frequency(control_rate)
    pid.pid_set_gains(kp, ki, kd)
    return pid


class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.declare_parameter("mujoco_xml_path")
        self.declare_parameter("sim_time_sec")
        self.declare_parameter("visualization_rate")
        self.declare_parameter("visualize_mujoco")
        self.visualize_mujoco = self.get_parameter("visualize_mujoco").get_parameter_value().bool_value
        mujoco_xml_path = self.get_parameter("mujoco_xml_path").get_parameter_value().string_value
        self.sim_time_sec = self.get_parameter("sim_time_sec").get_parameter_value().double_value
        self.visualization_rate = self.get_parameter("visualization_rate").get_parameter_value().double_value
        self.initialization_done = False
        self.goal_pos = [0.0, 0.0]
        self.contact_states = {'FR_FOOT': False,
                               'FL_FOOT': False}

        self.model = mj.MjModel.from_xml_path(mujoco_xml_path)
        mj.mj_printModel(self.model, 'robot_information.txt')
        self.data = mj.MjData(self.model)
        self.lock = Lock()

        self.time = 0
        self.model.opt.timestep = self.sim_time_sec
        self.dt = self.model.opt.timestep
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)

        self.odometry_base_pub = self.create_publisher(Odometry, 'odometry', 10)
        self.contact_right_pub = self.create_publisher(StampedBool, '~/contact_foot_right', 10)
        self.contact_left_pub = self.create_publisher(StampedBool, '~/contact_foot_left', 10)

        self.joint_states_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.joint_traj_sub = self.create_subscription(JointTrajectory, 'joint_trajectory', self.joint_traj_cb, 10)
        self.joint_traj_msg = None
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.init_cb, 10)
        self.reset_sub = self.create_subscription(Empty, '~/reset', self.reset_cb, 10)

        self.paused = True
        self.step_sim_sub = self.create_subscription(Float64, "~/step", self.step_cb, 1)
        self.pause_sim_sub = self.create_subscription(Bool, "~/pause", self.pause_cb, 1)

        self.timer = self.create_timer(self.dt, self.timer_cb)

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
    
    def reset_cb(self, msg):
        with self.lock:
            self.init([0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

    def init_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self.init([p.x, p.y, p.z], q=[q.w, q.x, q.y, q.z])

    def init(self, p, q=[1.0, 0.0, 0.0, 0.0]):
        self.model.eq_data[0][0] = -p[0]
        self.model.eq_data[0][1] = -p[1]
        self.model.eq_data[0][2] = -1.5 # one meter above gnd

        self.data.qpos = [0.0] * self.model.nq
        self.data.qpos[3] = q[0]
        self.data.qpos[4] = q[1]
        self.data.qpos[5] = q[2]
        self.data.qpos[6] = q[3]

        self.data.qvel = [0.0]* self.model.nv

        self.data.qpos[0] = p[0]
        self.data.qpos[1] = p[1]
        self.data.qpos[2] = -self.model.eq_data[0][2]

        self.model.eq_active = 1
        self.initialization_done = False

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
            self.model.eq_data[0][2] += 0.5 * self.dt

        if self.contact_states['FR_FOOT'] or self.contact_states['FL_FOOT']:
            if not self.initialization_done:
                self.initialization_done = True
                self.data.qvel = [0.0]* self.model.nv
                self.model.eq_active = 0 # let go of the robot
        
        if self.visualize_mujoco is True:
            vis_update_downsampling = int(round(1.0/self.visualization_rate/self.sim_time_sec/10))
            if self.counter % vis_update_downsampling == 0:
                self.viewer.render()

        self.run_joint_controllers()
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

        R_b_to_I = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        v_b = R_b_to_I.T @ self.data.qvel[0:3] # linear vel is in inertial frame
        msg_odom.twist.twist.linear.x = v_b[0]
        msg_odom.twist.twist.linear.y = v_b[1]
        msg_odom.twist.twist.linear.z = v_b[2]
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
        msg_contact_right.data = self.contact_states['FR_FOOT']
        msg_contact_left.header.stamp.sec = int(self.time)
        msg_contact_left.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_contact_left.data = self.contact_states['FL_FOOT']
        self.contact_right_pub.publish(msg_contact_right)
        self.contact_left_pub.publish(msg_contact_left)

        msg_joint_states = JointState()
        msg_joint_states.header.stamp.sec = int(self.time)
        msg_joint_states.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]

            msg_joint_states.name.append(key)
            msg_joint_states.position.append(value['actual_pos'])
            msg_joint_states.velocity.append(value['actual_vel'])
        self.joint_states_pub.publish(msg_joint_states)

        
        self.ankle_foot_spring('FL_ANKLE')
        self.ankle_foot_spring('FR_ANKLE')

    def run_joint_controllers(self):
        if self.joint_traj_msg is None:
            return
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            id_joint_msg = self.joint_traj_msg.joint_names.index(key)
            value['desired_pos'] = self.joint_traj_msg.points[0].positions[id_joint_msg]
            if self.joint_traj_msg.points[0].velocities:
                value['desired_vel'] = self.joint_traj_msg.points[0].velocities[id_joint_msg]

        Kp = 2*15.0*np.ones(self.model.njnt - 1) # exclude root
        Kp[1] *= 4
        Kp[6] *= 4

        i = 0
        for key, value in self.q_joints.items():
            if key != 'FL_ANKLE' and key != 'FR_ANKLE':
                error = value['actual_pos'] - value['desired_pos']
                actuators_torque = -Kp[i]*error
                actuators_vel = value['desired_vel']
                self.data.ctrl[self.q_actuator_addr[str(key)]] = actuators_torque
                self.data.ctrl[self.q_actuator_addr[str(key) + "_VEL"]] = actuators_vel
            i = i + 1

    # def stop_controller(self, actuator_name):
    #     idx_act = mj.mj_name2id(
    #         self.model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    #     self.data.ctrl[idx_act] = 0.0

    def joint_traj_cb(self, msg):
        with self.lock:
            self.joint_traj_msg = msg

    def read_contact_states(self):
        self.contact_states['FR_FOOT'] = False
        self.contact_states['FL_FOOT'] = False

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
            if 'FL_FOOT' in geom2_list:
                first_entry_idx = geom2_list.index('FL_FOOT')
                if geom1_list[first_entry_idx] == 'floor':
                    self.contact_states['FL_FOOT'] = True

            if 'FR_FOOT' in geom2_list:
                first_entry_idx = geom2_list.index('FR_FOOT')
                if geom1_list[first_entry_idx] == 'floor':
                    self.contact_states['FR_FOOT'] = True

    def get_joint_names(self):
        self.name_joints = []
        for i in range(1, self.model.njnt):  # skip root
            self.name_joints.append(mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_JOINT, i))
        return self.name_joints


    def ankle_foot_spring(self, foot_joint):
        'ankle modelled as a spring damped system'
        K = 0.0003
        offset = -0.5
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
