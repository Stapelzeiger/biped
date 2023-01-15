import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory, JointTrajectory
from geometry_msgs.msg import TransformStamped, Vector3

from rosgraph_msgs.msg import Clock
from linux_gpio.msg import StampedBool
from std_msgs.msg import Bool
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


def setup_pid(control_rate, kp, ki, kd):
    pid = pid_ctrl()
    pid.pid_set_frequency(control_rate)
    pid.pid_set_gains(kp, ki, kd)
    return pid


class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        
        self.declare_parameter("mujoco_xml_path")
        mujoco_xml_path = self.get_parameter("mujoco_xml_path").get_parameter_value().string_value

        self.declare_parameter("sim_time_sec")
        self.sim_time_sec = self.get_parameter("sim_time_sec").get_parameter_value().double_value

        self.declare_parameter("visualization_rate")
        self.visualization_rate = self.get_parameter("visualization_rate").get_parameter_value().double_value

        self.model = mj.MjModel.from_xml_path(mujoco_xml_path)
        mj.mj_printModel(self.model, 'robot_information.txt')
        self.data = mj.MjData(self.model)

        self.time = 0
        self.model.opt.timestep = self.sim_time_sec
        self.dt = self.model.opt.timestep
        self.odometry_base_pub = self.create_publisher(Odometry, 'odometry', 10)
        
        self.joint_traj_sub = self.create_subscription(JointTrajectory, 'joint_trajectory', self.joint_traj_cb, 10)
        self.joint_traj_sub  # prevent unused variable warning

        self.initialization_flag_sub = self.create_subscription(Bool, '/initialization_flag', self.initialization_flag_cb, 10)
        self.initialization_flag_sub
        self.initialization_flag = False

        self.next_footstep_OdomF_sub = self.create_subscription(Vector3, 'next_footstep_OdomF', self.next_footstep_OdomF_cb, 10)
        self.next_footstep_OdomF_sub

        self.desired_swing_foot_OdomF_sub = self.create_subscription(Vector3, 'desired_swing_foot_pos_OdomF', self.desired_swing_foot_OdomF_cb, 10)
        self.desired_swing_foot_OdomF_sub

        self.joint_states_pub = self.create_publisher(JointState, 'joint_states', 10)

        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.timer = self.create_timer(self.dt, self.step)

        self.nb_joints = self.model.njnt - 1 # exclude root
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

        self.contact_pub = self.create_publisher(StampedBool, '/contact', 10)

        self.contact_states = {'FR_FOOT': False,
                               'FL_FOOT': False}

        self.declare_parameter("visualize_mujoco")
        self.visualize_mujoco = self.get_parameter("visualize_mujoco").get_parameter_value().bool_value

        if self.visualize_mujoco == True:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -25
            self.viewer.render()

        self.q_pos_addr_joints = {}
        for name in self.name_joints:
            self.q_pos_addr_joints[name] = self.model.jnt_qposadr[mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, name)]
        
        self.pid_pitch_foot = setup_pid(
            control_rate=1.0/self.dt, kp=0.002, ki=0.0, kd=0)

        self.counter = 0
        self.desired_swing_foot_OdomF = [0.0, 0.0, 0.0]
        self.desired_swing_foot_OdomF_list = []

    def initialization_flag_cb(self, msg):
        self.initialization_flag = msg.data
        print(self.initialization_flag)

    def step(self):
        self.time += self.dt
        self.counter += 1

        if self.initialization_flag == True:
            self.model.eq_active = 0
        
        if self.visualize_mujoco is True:
            vis_update_downsampling = int(round(1.0/self.visualization_rate/self.sim_time_sec/10))
            if self.counter % vis_update_downsampling == 0:
                self.viewer.render()

        self.viewer.render()
        mj.mj_step(self.model, self.data)

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

        msg_odom.twist.twist.linear.x = self.data.qvel[0]
        msg_odom.twist.twist.linear.y = self.data.qvel[1]
        msg_odom.twist.twist.linear.z = self.data.qvel[2]
        msg_odom.twist.twist.angular.x = self.data.qvel[3]
        msg_odom.twist.twist.angular.y = self.data.qvel[4]
        msg_odom.twist.twist.angular.z = self.data.qvel[5]
        self.odometry_base_pub.publish(msg_odom)

        self.read_contact_states()
        msg_contact = StampedBool()
        msg_contact.header.stamp.sec = int(self.time)
        msg_contact.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_contact.data = list(self.contact_states.values())
        msg_contact.names = list(self.contact_states.keys())
        self.contact_pub.publish(msg_contact)

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


        if self.contact_states['FR_FOOT'] == True:
            self.keep_ankle_foot_horiz_with_gnd_controller('FL_ANKLE')
        if self.contact_states['FL_FOOT'] == True:
            self.keep_ankle_foot_horiz_with_gnd_controller('FR_ANKLE')

        # self.desired_swing_foot_OdomF_list.append([self.desired_swing_foot_OdomF[0], self.desired_swing_foot_OdomF[1], self.desired_swing_foot_OdomF[2]])
        # for l in self.desired_swing_foot_OdomF_list:
        #     self.viewer.add_marker(pos=[l[0], l[1], l[2]], size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 1], type=mj.mjtGeom.mjGEOM_SPHERE, label="f")



    def stop_controller(self, actuator_name):
        idx_act = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        self.data.ctrl[idx_act] = 0.0

    def joint_traj_cb(self, msg):
    
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            id_joint_msg = msg.joint_names.index(key)
            value['desired_pos'] = msg.points[0].positions[id_joint_msg]
            if msg.points[0].velocities:
                value['desired_vel'] = msg.points[0].velocities[id_joint_msg]

        Kp = 2*15.0*np.ones(self.nb_joints)
        Kp[1] *= 6
        Kp[6] *= 6 

        i = 0
        for key, value in self.q_joints.items():
            if key != 'FL_ANKLE' and key != 'FR_ANKLE':
                error = value['actual_pos'] - value['desired_pos']
                print('error for joint', key, 'is', error)
                actuators_torque = -Kp[i]*error
                actuators_vel = value['desired_vel']
                self.data.ctrl[self.q_actuator_addr[str(key)]] = actuators_torque
                self.data.ctrl[self.q_actuator_addr[str(key) + "_VEL"]] = actuators_vel
            i = i + 1

    def next_footstep_OdomF_cb(self, msg):
        pos = [msg.x, msg.y, msg.z]
        self.viewer.add_marker(pos=[pos[0], pos[1], 0], size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 1], type=mj.mjtGeom.mjGEOM_SPHERE, label="next_footstep")

    def desired_swing_foot_OdomF_cb(self, msg):
        self.desired_swing_foot_OdomF = [msg.x, msg.y, msg.z]

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


    def keep_ankle_foot_horiz_with_gnd_controller(self, swing_foot_joint_name):
        'ankle modelled as a spring damped system'

        if swing_foot_joint_name == 'FL_ANKLE':
            idx_body = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_BODY, 'FL_FOOT')
            idx_act_stance = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'FR_ANKLE')
            idx_act_vel_stance = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'FR_ANKLE_VEL')

        else:
            idx_body = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_BODY, 'FR_FOOT')
            idx_act_stance = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'FL_ANKLE')
            idx_act_vel_stance = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'FL_ANKLE_VEL')

        foot_body_quat = self.data.xquat[idx_body]
        r = R.from_quat([foot_body_quat[1], foot_body_quat[2],
                        foot_body_quat[3], foot_body_quat[0]])  # convention x,y,z,w
        pitch_foot_body = r.as_euler('ZYX', degrees=False)[1] # this is not really working well for angles that are > 90 deg

        pitch_error_foot = pitch_foot_body - np.deg2rad(0.0)
        pitch_torque_setpt = self.pid_pitch_foot.pid_process(pitch_error_foot)

        if abs(np.rad2deg(pitch_foot_body)) > 45:
            print('FOOOT OVERBOARD!!!')
            pitch_torque_setpt = 0.0
            self.data.qpos[self.q_pos_addr_joints[swing_foot_joint_name]] = 0.0

        actuator_name = swing_foot_joint_name
        idx_act = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        vel_actuator_name = str(swing_foot_joint_name) + str('_VEL')
        idx_vel_act = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_ACTUATOR, vel_actuator_name)

        self.data.ctrl[idx_act] = pitch_torque_setpt
        self.data.ctrl[idx_vel_act] = 0.0
        self.data.ctrl[idx_act_stance] = 0.0
        self.data.ctrl[idx_act_vel_stance] = 0.0




def main(args=None):
    rclpy.init(args=args)
    sim_node = MujocoNode()
    rclpy.spin(sim_node)
    sim_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
