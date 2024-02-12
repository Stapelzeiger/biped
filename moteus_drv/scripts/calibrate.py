#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

class JointTrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('joint_publisher')
        self.publisher_ = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.timer_period = 0.01  # seconds
        self.t = 0.0
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.joint_names = ['R_YAW', 'R_HAA', 'R_HFE', 'R_KFE', 'L_YAW', 'L_HAA', 'L_HFE', 'L_KFE']

    def timer_callback(self):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        r_kfe_traj = np.sin(2*np.pi*self.t)
        l_kfe_traj = np.cos(2*np.pi*self.t)
        r_kfe_vel = 2*np.pi*np.cos(2*np.pi*self.t)
        l_kfe_vel = -2*np.pi*np.sin(2*np.pi*self.t)
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, r_kfe_traj, 0.0, 0.0, 0.0, l_kfe_traj]
        point.velocities = [0.0, 0.0, 0.0, r_kfe_vel, 0.0, 0.0, 0.0, l_kfe_vel]
        msg.points.append(point)
        self.publisher_.publish(msg)
        self.t += self.timer_period
        self.get_logger().info('Published joint trajectory')

def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointTrajectoryPublisher()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()