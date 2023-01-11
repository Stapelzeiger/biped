import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory, JointTrajectory
from geometry_msgs.msg import Transform

from rosgraph_msgs.msg import Clock
from linux_gpio.msg import StampedBool

import mujoco as mj
import mujoco_viewer
import numpy as np
import sys
import json

class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        
        self.declare_parameter("mujoco_xml_path")
        mujoco_xml_path = self.get_parameter("mujoco_xml_path").get_parameter_value().string_value

        self.model = mj.MjModel.from_xml_path(mujoco_xml_path)
        mj.mj_printModel(self.model, 'robot_information.txt')
        self.data = mj.MjData(self.model)

        self.time = 0
        self.dt = self.model.opt.timestep
        self.odometry_base_pub = self.create_publisher(Odometry, 'odometry', 10)

        self.joint_traj_sub = self.create_subscription(JointTrajectory, 'joint_trajectory', self.joint_traj_cb, 10)
        self.joint_traj_sub  # prevent unused variable warning

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

        self.right_foot_contact_pub = self.create_publisher(StampedBool, 'right_foot_contact', 10)
        self.left_foot_contact_pub = self.create_publisher(StampedBool, 'left_foot_contact', 10)
        self.contact_states = {'FR_ANKLE': False,
                               'FL_ANKLE': False,
                               'both': False}

        self.declare_parameter("visualize_mujoco")
        self.visualize_mujoco = self.get_parameter("visualize_mujoco").get_parameter_value().bool_value

        if self.visualize_mujoco == True:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -25
            self.viewer.render()

    def step(self):
        self.time += self.dt
        if self.visualize_mujoco == True:
            self.viewer.render()
        mj.mj_step(self.model, self.data)

        clock_msg = Clock()
        clock_msg.clock.sec = int(self.time)
        clock_msg.clock.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        self.clock_pub.publish(clock_msg)

        msg_odom = Odometry()
        msg_odom.header.stamp.sec = int(self.time)
        msg_odom.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
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
        msg_foot_sensor_right = StampedBool()
        msg_foot_sensor_right.header.stamp.sec = int(self.time)
        msg_foot_sensor_right.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_foot_sensor_right.data = self.contact_states['FR_ANKLE']
        self.right_foot_contact_pub.publish(msg_foot_sensor_right)

        msg_foot_sensor_left = StampedBool()
        msg_foot_sensor_left.header.stamp.sec = int(self.time)
        msg_foot_sensor_left.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg_foot_sensor_left.data = self.contact_states['FL_ANKLE']
        self.left_foot_contact_pub.publish(msg_foot_sensor_left)


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
        Kp[1] *= 2 
        Kp[6] *= 2 

        i = 0
        for key, value in self.q_joints.items():
            if key != 'FL_ANKLE' and key != 'FR_ANKLE':
                actuators_torque = -Kp[i]*(value['actual_pos'] - value['desired_pos'])
                actuators_vel = value['desired_vel']
                self.data.ctrl[self.q_actuator_addr[str(key)]] = actuators_torque
                self.data.ctrl[self.q_actuator_addr[str(key) + "_VEL"]] = actuators_vel
            i = i + 1


    def reset_contact_state(self):
        self.contact_states['FR_ANKLE'] = False
        self.contact_states['FL_ANKLE'] = False
        self.contact_states['both'] = False

    def read_contact_states(self):
        self.reset_contact_state()

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
                    self.contact_states['FL_ANKLE'] = True

            if 'FR_FOOT' in geom2_list:
                first_entry_idx = geom2_list.index('FR_FOOT')
                if geom1_list[first_entry_idx] == 'floor':
                    self.contact_states['FR_ANKLE'] = True

            if 'FR_FOOT' in geom2_list and 'FL_FOOT' in geom2_list:
                self.contact_states['both'] = True


    def get_joint_names(self):
        self.name_joints = []
        for i in range(1, self.model.njnt):  # skip root
            self.name_joints.append(mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_JOINT, i))
        return self.name_joints

def main(args=None):
    rclpy.init(args=args)
    sim_node = MujocoNode()
    rclpy.spin(sim_node)
    sim_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
