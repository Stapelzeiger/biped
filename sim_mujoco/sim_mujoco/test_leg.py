import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped

from rosgraph_msgs.msg import Clock
from biped_bringup.msg import StampedBool
from std_msgs.msg import Bool, Float64, Empty
from tf2_ros import TransformBroadcaster, TransformException
from scipy.spatial.transform import Rotation as R

import mujoco.viewer
import mujoco as mj
import numpy as np
import time
from sim_mujoco.submodules.pid import pid as pid_ctrl
from threading import Lock
import math 


def setup_pid(control_rate, kp, ki, kd):
    pid = pid_ctrl()
    pid.pid_set_frequency(control_rate)
    pid.pid_set_gains(kp, ki, kd)
    return pid

class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.get_logger().info("Start Sim!")
        self.declare_parameter("mujoco_xml_path", rclpy.parameter.Parameter.Type.STRING)
        self.declare_parameter("sim_time_sec", rclpy.parameter.Parameter.Type.DOUBLE)
        self.declare_parameter("visualize_mujoco", rclpy.parameter.Parameter.Type.BOOL)
        self.declare_parameter("publish_tf", rclpy.parameter.Parameter.Type.BOOL)
        self.visualize_mujoco = self.get_parameter("visualize_mujoco").get_parameter_value().bool_value
        mujoco_xml_path = self.get_parameter("mujoco_xml_path").get_parameter_value().string_value
        self.sim_time_sec = self.get_parameter("sim_time_sec").get_parameter_value().double_value
        self.publish_tf = self.get_parameter("publish_tf").get_parameter_value().bool_value
        self.initialization_done = False
        self.goal_pos = [0.0, 0.0]
        self.contact_states = {'L_FOOT': False}

        self.model = mj.MjModel.from_xml_path(mujoco_xml_path)
        self.data = mj.MjData(self.model)

        if self.visualize_mujoco is True:
            self.get_logger().info("Start visualization!")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        mj.mj_printModel(self.model, 'robot_information.txt')

        self.lock = Lock()

        self.time = time.time()
        self.model.opt.timestep = self.sim_time_sec
        self.dt = self.model.opt.timestep

        self.name_joints = self.get_joint_names()

        self.q_joints = {}
        for i in self.name_joints:
            self.q_joints[i] = {
                'timestamp': 0.0,
                'actual_pos': 0.0,
                'actual_vel': 0.0,
                'actual_acc': 0.0,
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
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)

        self.contact_right_pub = self.create_publisher(StampedBool, '~/contact_foot_right', 10)
        self.contact_left_pub = self.create_publisher(StampedBool, '~/contact_foot_left', 10)

        self.stop_pub = self.create_publisher(Bool, '~/stop', 10)

        self.joint_states_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '~/imu', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.joint_traj_sub = self.create_subscription(JointTrajectory, 'joint_trajectory', self.joint_traj_cb, 10)
        self.joint_traj_msg = None

        self.paused = False
        self.step_sim_sub = self.create_subscription(Float64, "~/step", self.step_cb, 1)
        self.pause_sim_sub = self.create_subscription(Bool, "~/pause", self.pause_cb, 1)
        self.previous_q_vel = np.zeros(self.model.nv)
        self.timer = self.create_timer(self.dt*2, self.timer_cb, clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME))

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

    def stop_viz(self):
        self.viewer.close()

    def step(self):

        self.read_contact_states()

        for _ in range(2):
            self.run_joint_controllers()
            self.ankle_foot_spring('L_ANKLE')
            mj.mj_step(self.model, self.data)
            if self.visualize_mujoco is True:
                if self.viewer.is_running():
                    self.viewer.sync()
            self.time += self.dt
            self.counter += 1

        clock_msg = Clock()
        clock_msg.clock.sec = int(self.time)
        clock_msg.clock.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        self.clock_pub.publish(clock_msg)

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
            msg_joint_states.name.append(key)
            msg_joint_states.position.append(value['actual_pos'])
            msg_joint_states.velocity.append(value['actual_vel'])
            msg_joint_states.effort.append(value['actual_acc'])
        self.joint_states_pub.publish(msg_joint_states)


    def run_joint_controllers(self):
        if self.joint_traj_msg is None:
            return
        for key, value in self.q_joints.items():
            id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
            value['actual_pos'] = self.data.qpos[self.model.jnt_qposadr[id_joint_mj]]
            value['actual_vel'] = self.data.qvel[self.model.jnt_dofadr[id_joint_mj]]
            actual_acc = (self.data.qvel[self.model.jnt_dofadr[id_joint_mj]] - self.previous_q_vel[self.model.jnt_dofadr[id_joint_mj]])/self.dt
            value['actual_acc'] = actual_acc
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
        self.previous_q_vel = self.data.qvel.copy()

        kp_moteus = 240.0
        Kp = (kp_moteus/(2*math.pi)) * np.ones(self.model.njnt - 1) # exclude root

        i = 0
        for key, value in self.q_joints.items():
            if key != 'L_ANKLE':
                error = value['actual_pos'] - value['desired_pos']
                actuators_torque = - Kp[i]*error
                actuators_vel = value['desired_vel']
                feedforward_torque = value['feedforward_torque']
                self.data.ctrl[self.q_actuator_addr[str(key)]] = actuators_torque
                self.data.ctrl[self.q_actuator_addr[str(key) + "_VEL"]] = actuators_vel
                id_joint_mj = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, key)
                self.data.qfrc_applied[self.model.jnt_dofadr[id_joint_mj]] = feedforward_torque
            i = i + 1

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
    sim_node.stop_viz()
    sim_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
