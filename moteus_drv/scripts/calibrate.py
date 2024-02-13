#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import threading

class JointCalibration(Node):
    def __init__(self):
        super().__init__('joint_calibration_publisher')
        self.pub_trajectory = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.joint_states_sub = self.create_subscription(JointTrajectory, 'joint_states', self.joint_states_callback, 10)
        self.timer_period = 0.01 # seconds

        self.joints_dictionary = { # TODO populate this dict from the config params.yaml
            'joint_names': ['test_motor'],
            'is_calibrated': [False],
            'joint_pos': [],
            'joint_vel': [],
            'center_pos': [],
            'upper_limit': [],
            'lower_limit': [],
        }
        self.calibrated_joints = []
        self.counter = 0
        self.all_motors_calibrated = False
        self.velocity_max = 0.01
        self.joint_states = None
        self.lock = threading.Lock()
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        # self.joint_names = ['R_YAW', 'R_HAA', 'R_HFE', 'R_KFE', 'L_YAW', 'L_HAA', 'L_HFE', 'L_KFE']

    def joint_states_callback(self, msg):
        with self.lock:
            for i, joint in enumerate(self.joints_dictionary['joint_names']):
                if joint in msg.joint_names:
                    idx = msg.joint_names.index(joint)
                    self.joints_dictionary['joint_pos'].append(msg.points[idx].positions[0])
                    self.joints_dictionary['joint_vel'].append(msg.points[idx].velocities[0])

    def calibrate_motor(self, joint, idx):
        setpt_pos = self.velocity_max*self.counter
        setpt_vel = self.velocity_max

        joint_vel = self.joints_dictionary['joint_vel'][idx]
        if (np.abs(setpt_vel - joint_vel) > 0.01 and self.velocity_max > 0): # going forward
            self.get_logger().info(f'Joint {joint} is not moving, record position')
            self.joints_dictionary['upper_limit'].append(self.joints_dictionary['joint_pos'][idx])
            print('Upper Limit', self.joints_dictionary['upper_limit'])
            self.velocity_max = -self.velocity_max
            self.counter = 0

        if (np.abs(setpt_vel - joint_vel) > 0.01 and self.velocity_max < 0): # going backward
            self.get_logger().info(f'Joint {joint} is not moving, record position')
            self.joints_dictionary['lower_limit'].append(self.joints_dictionary['joint_pos'][idx])
            print('Lower Limit', self.joints_dictionary['lower_limit'])
            self.velocity_max = -self.velocity_max
            self.joints_dictionary['is_calibrated'][idx] = True
            self.joints_dictionary['center_pos'].append((self.joints_dictionary['upper_limit'][idx] + self.joints_dictionary['lower_limit'][idx])/2)
            self.counter = 0

        msg = JointTrajectory()
        msg.joint_names = joint
        setpt_pos = self.velocity_max*self.counter
        setpt_vel = self.velocity_max
        point = JointTrajectoryPoint()
        point.positions = [setpt_pos]
        point.velocities = [setpt_vel]
        msg.points.append(point)
        self.pub_trajectory.publish(msg)
        self.get_logger().info(f'Published joint trajectory for {joint} with setpt_pos: {setpt_pos} and setpt_vel: {setpt_vel}')

    def timer_callback(self):
        if self.joints_dictionary['is_calibrated'] == [True]*len(self.joints_dictionary['joint_names']):
            self.all_motors_calibrated = True
            self.get_logger().info('All motors are calibrated')
            return

        if self.joints_dictionary['joint_states'] == []:
            self.get_logger().info('No joint states received yet')
            return

        # take the first uncalibrated joint and calibrate it
        for i, joint in enumerate(self.joints_dictionary['joint_names']):
            if self.joints_dictionary['is_calibrated'][i] == False:
                self.get_logger().info(f'Calibrating joint {joint}')
                self.calibrate_motor(joint, i)
                break

        self.counter += self.timer_period
        self.get_logger().info('Published joint trajectory')

def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointCalibration()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()