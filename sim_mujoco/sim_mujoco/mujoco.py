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


class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        
        self.declare_parameter("mujoco_xml_path")
        mujoco_xml_path = self.get_parameter("mujoco_xml_path").get_parameter_value().string_value
        self.model = mj.MjModel.from_xml_path(mujoco_xml_path)
        self.data = mj.MjData(self.model)

        self.time = 0
        self.dt = self.model.opt.timestep
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.odometry_base_pub = self.create_publisher(Odometry, 'odom', 10)
        self.foot_position_BL_pub = self.create_publisher(MultiDOFJointTrajectory, 'foot_position_BL', 10)

        self.joint_traj_sub = self.create_subscription(JointTrajectory, 'joint_traj', self.joint_traj_cb, 10)
        self.joint_traj_sub  # prevent unused variable warning

        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.timer = self.create_timer(self.dt, self.step)

        self.declare_parameter("visualize_mujoco")
        self.visualize_mujoco = self.get_parameter("visualize_mujoco").get_parameter_value().bool_value
        print('------>', self.visualize_mujoco)
        if self.visualize_mujoco == True:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -25
            self.viewer.render()

        self.name_joints = self.get_joint_names()
        print(self.name_joints)        

    def step(self):
        self.time += self.dt
        if self.visualize_mujoco == True:
            self.viewer.render()
        mj.mj_step(self.model, self.data)

        clock_msg = Clock()
        clock_msg.clock.sec = int(self.time)
        clock_msg.clock.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        self.clock_pub.publish(clock_msg)

        msg = JointState()
        msg.header.stamp.sec = int(self.time)
        msg.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        msg.name = self.name_joints
        msg.position = list(self.data.qpos.copy()[self.model.jnt_qposadr[1]: ]) # skip root
        msg.velocity = list(self.data.qvel.copy()[self.model.jnt_dofadr[1]: ]) # skip root
        self.joint_state_pub.publish(msg)

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


        des_p_stance_foot_BLF = np.array([0.0, 0.0, -0.58])
        name_stance_foot = "FL_ANKLE"
        transforms = Transform()
        transforms.translation.x = des_p_stance_foot_BLF[0]
        transforms.translation.y = des_p_stance_foot_BLF[1]
        transforms.translation.z = des_p_stance_foot_BLF[2]
        transforms.rotation.x = 0.0; transforms.rotation.y = 0.0
        transforms.rotation.z = 0.0; transforms.rotation.w = 1.0

        foot_pos_point = MultiDOFJointTrajectoryPoint()
        foot_pos_point.transforms = [transforms]

        msg_foot = MultiDOFJointTrajectory()
        msg_foot.joint_names = [name_stance_foot]
        msg_foot.points = [foot_pos_point]
        self.foot_position_BL_pub.publish(msg_foot)


    def joint_traj_cb(self, msg):
        print(msg.joint_names)


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
