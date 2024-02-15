#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np
import threading

class JointCalibration(Node):
    def __init__(self):
        super().__init__('joint_calibration_publisher')
        self.pub_trajectory = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.joint_states_sub = self.create_subscription(JointState, 'joint_states', self.joint_states_callback, 10)
        self.timer_period = 0.01 # seconds

        self.joints_dictionary = { # TODO populate this dict from the config params.yaml
            'joint_names': ['test1'],
            'is_calibrated': [False],
            'joint_pos': [None],
            'joint_vel': [None],
            'joint_effort': [None],
            'center_pos': [None],
            'upper_limit': [None],
            'lower_limit': [None],
            'initial_pos': [None],
        }
        self.calibrated_joints = []
        self.counter = 0
        self.all_motors_calibrated = False
        self.velocity_max = 0.1
        self.joint_states = None
        self.lock = threading.Lock()
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        # self.joint_names = ['R_YAW', 'R_HAA', 'R_HFE', 'R_KFE', 'L_YAW', 'L_HAA', 'L_HFE', 'L_KFE']

    def joint_states_callback(self, msg):
        with self.lock:
            for i, joint in enumerate(self.joints_dictionary['joint_names']):
                if joint in msg.name:
                    idx = msg.name.index(joint)
                    self.joints_dictionary['joint_pos'][idx] = msg.position[idx]
                    self.joints_dictionary['joint_vel'][idx] = msg.velocity[idx]
                    self.joints_dictionary['joint_effort'][idx] = msg.effort[idx]
                    self.joints_dictionary['initial_pos'][idx] = msg.position[idx]

    def calibrate_motor(self, joint, idx):
        setpt_pos = self.velocity_max*self.counter/1000
        setpt_vel = self.velocity_max

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [joint]
        msg.points = [JointTrajectoryPoint()]
        msg.points[0].positions.append(setpt_pos + self.joints_dictionary['initial_pos'][idx])
        msg.points[0].velocities.append(setpt_vel)
        msg.points[0].effort.append(0)
        self.pub_trajectory.publish(msg)
        max_trigger_effort = 1.5

        joint_vel = self.joints_dictionary['joint_vel'][idx]
        joint_effort = self.joints_dictionary['joint_effort'][idx]
        if np.abs(joint_effort) > max_trigger_effort:
            self.get_logger().info('joint effort triggered')
        if (np.abs(joint_effort) > max_trigger_effort and self.velocity_max > 0 and self.counter > 20): # going forward
            self.get_logger().info(f'Joint {joint} is not moving, record position')
            self.joints_dictionary['upper_limit'][idx] = self.joints_dictionary['joint_pos'][idx]
            print('Upper Limit', self.joints_dictionary['upper_limit'])
            self.velocity_max = -self.velocity_max
            self.counter = 0

        if (np.abs(joint_effort) > max_trigger_effort and self.velocity_max < 0 and self.counter > 20): # going backward
            self.get_logger().info(f'Joint {joint} is not moving, record position')
            self.joints_dictionary['lower_limit'][idx] = self.joints_dictionary['joint_pos'][idx]
            print('Lower Limit', self.joints_dictionary['lower_limit'])
            self.velocity_max = -self.velocity_max
            # goto center position
            self.joints_dictionary['is_calibrated'][idx] = True
            self.joints_dictionary['center_pos'][idx] = (self.joints_dictionary['upper_limit'][idx] + self.joints_dictionary['lower_limit'][idx])/2
            self.counter = 0

    def timer_callback(self):
        if self.joints_dictionary['is_calibrated'] == [True]*len(self.joints_dictionary['joint_names']):
            self.all_motors_calibrated = True
            self.get_logger().info('All motors are calibrated')
            # exit ros
            rclpy.shutdown()
            return

        if None in self.joints_dictionary['joint_pos'] or \
            None in self.joints_dictionary['joint_vel'] or \
            None in self.joints_dictionary['joint_effort'] or \
            None in self.joints_dictionary['initial_pos']:
            self.get_logger().info('No joint states received yet')
            return

        # take the first uncalibrated joint and calibrate it
        for i, joint in enumerate(self.joints_dictionary['joint_names']):
            if self.joints_dictionary['is_calibrated'][i] == False:
                # self.get_logger().info(f'Calibrating joint {joint}')
                self.calibrate_motor(joint, i)
                break

        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointCalibration()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()