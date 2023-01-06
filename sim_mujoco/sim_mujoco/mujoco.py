import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory, JointTrajectory
from geometry_msgs.msg import Transform

from rosgraph_msgs.msg import Clock

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
        self.odometry_base_pub = self.create_publisher(Odometry, 'odom', 10)

        self.joint_traj_sub = self.create_subscription(JointTrajectory, 'joint_trajectory', self.joint_traj_cb, 10)
        self.joint_traj_sub  # prevent unused variable warning

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
            self.name_actuators.append(mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, i))

        self.q_actuator_addr = {}
        for name in self.name_actuators:
            self.q_actuator_addr[name] = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)

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
        msg_odom.pose.pose.position.x = self.data.qpos.copy()[0]
        msg_odom.pose.pose.position.y = self.data.qpos.copy()[1]
        msg_odom.pose.pose.position.z = self.data.qpos.copy()[2]
        msg_odom.pose.pose.orientation.w = self.data.qpos.copy()[3]
        msg_odom.pose.pose.orientation.x = self.data.qpos.copy()[4]
        msg_odom.pose.pose.orientation.y = self.data.qpos.copy()[5]
        msg_odom.pose.pose.orientation.z = self.data.qpos.copy()[6]
        self.odometry_base_pub.publish(msg_odom)

    def set_control_input(self, actuators_torque, actuators_vel):
        self.data.ctrl[self.q_actuator_addr["FL_YAW"]] = actuators_torque[0]
        self.data.ctrl[self.q_actuator_addr["FL_YAW_VEL"]] = actuators_vel[0]

        self.data.ctrl[self.q_actuator_addr["FL_HAA"]] = actuators_torque[1]
        self.data.ctrl[self.q_actuator_addr["FL_HAA_VEL"]] = actuators_vel[1]
        self.data.ctrl[self.q_actuator_addr["FL_HFE"]] = actuators_torque[2]
        self.data.ctrl[self.q_actuator_addr["FL_HFE_VEL"]] = actuators_vel[2]
        self.data.ctrl[self.q_actuator_addr["FL_KFE"]] = actuators_torque[3]
        self.data.ctrl[self.q_actuator_addr["FL_KFE_VEL"]] = actuators_vel[3]

        self.data.ctrl[self.q_actuator_addr["FR_YAW"]] = actuators_torque[5]
        self.data.ctrl[self.q_actuator_addr["FR_YAW_VEL"]] = actuators_vel[5]
        self.data.ctrl[self.q_actuator_addr["FR_HAA"]] = actuators_torque[6]
        self.data.ctrl[self.q_actuator_addr["FR_HAA_VEL"]] = actuators_vel[6]
        self.data.ctrl[self.q_actuator_addr["FR_HFE"]] = actuators_torque[7]
        self.data.ctrl[self.q_actuator_addr["FR_HFE_VEL"]] = actuators_vel[7]
        self.data.ctrl[self.q_actuator_addr["FR_KFE"]] = actuators_torque[8]
        self.data.ctrl[self.q_actuator_addr["FR_KFE_VEL"]] = actuators_vel[8]


    def get_control_joint_inputs(self, q_current_joints, q_dot_current_joints, q_des_joints, q_des_dot_joints):
        '''
        q_des = [ q_L_HAA, q_L_HFE, q_L_KFE, q_L_ANKLE, q_R_HAA, q_R_HFE, q_R_KFE, q_R_ANKLE]
        q_des_dot = [q_L_HAA_dot, q_L_HFE_dot, q_L_KFE_dot, q_L_ANKLE_dot, q_R_HAA_dot, q_R_HFE_dot, q_R_KFE_dot, q_R_ANKLE]

        '''
        
        Kp = 2*15.0*np.eye(self.nb_joints)
        # Kp = 2*np.eye(self.nb_joints)

        Kp[1, 1] *= 2 
        Kp[6, 6] *= 2 

        actuation_matrix = np.eye(self.nb_joints)

        actuators_torque = -Kp@actuation_matrix@(q_current_joints - q_des_joints)
        actuators_vel = actuation_matrix@q_des_dot_joints

        print('actuators_torque', actuators_torque)
        return actuators_torque, actuators_vel

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

        actuators_torque = []
        actuators_vel = []
        i = 0
        for key, value in self.q_joints.items():
            actuators_torque.append(-Kp[i]*(value['actual_pos'] - value['desired_pos']))
            actuators_vel.append(value['desired_vel'])
            i = i + 1

        self.set_control_input(actuators_torque, actuators_vel)

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
