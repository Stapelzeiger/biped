#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np
import threading

TIME_PERIOD = 0.01
VEL_MAX = 0.5
MAX_TRIGGER_EFFORT = 1.0
COUNTER_TRIGGER = 20
EPSILON = 0.001

class JointCalibration(Node):
    def __init__(self):
        super().__init__('joint_calibration_publisher')

        self.pub_trajectory = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.joint_states_sub = self.create_subscription(JointState, 'joint_states', self.joint_states_callback, 10)
        self.timer_period = TIME_PERIOD # seconds

        # self.joint_names = ['R_YAW', 'R_HAA', 'R_HFE', 'R_KFE', 'L_YAW', 'L_HAA', 'L_HFE', 'L_KFE']
        list_motors = self.declare_parameter('joints', rclpy.Parameter.Type.STRING_ARRAY).value
        self.joints_dictionary = { # TODO populate this dict from the config params.yaml
            'joint_names': list_motors,
            'is_calibrated': [False]*len(list_motors),
            'joint_pos': [None]*len(list_motors),
            'joint_vel': [None]*len(list_motors),
            'joint_effort': [None]*len(list_motors),
            'center_pos': [None]*len(list_motors),
            'upper_limit': [None]*len(list_motors),
            'lower_limit': [None]*len(list_motors),
            'initial_pos': [None]*len(list_motors)
        }

        print('Calibrating the following joints:')
        print(self.joints_dictionary)
        self.lock = threading.Lock()

        self.counter = 0
        self.counter_ramp_center = 0
        self.all_motors_calibrated = False

        self.velocity_max = VEL_MAX
        if self.velocity_max < 0:
            self.get_logger().info('Starting velocity max cannot be negative')
            return

        self.setpt_pos = self.velocity_max*self.counter/1000
        self.setpt_vel = self.velocity_max

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def joint_states_callback(self, msg):
        with self.lock:
            for _, joint in enumerate(self.joints_dictionary['joint_names']):
                if joint in msg.name:
                    idx = msg.name.index(joint)
                    self.joints_dictionary['joint_pos'][idx] = msg.position[idx]
                    self.joints_dictionary['joint_vel'][idx] = msg.velocity[idx]
                    self.joints_dictionary['joint_effort'][idx] = msg.effort[idx]
                    self.joints_dictionary['initial_pos'][idx] = msg.position[idx]

    def calibrate_motor(self, joint, idx):
        with self.lock:
            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = [joint]
            msg.points = [JointTrajectoryPoint()]
            msg.points[0].positions.append(self.setpt_pos)
            msg.points[0].velocities.append(self.setpt_vel)
            msg.points[0].effort.append(0)
            self.pub_trajectory.publish(msg)

            joint_effort = self.joints_dictionary['joint_effort'][idx]
            # if the joint is not moving and the center pos is not determined, record the position
            if np.abs(joint_effort) > MAX_TRIGGER_EFFORT and self.joints_dictionary['center_pos'][idx] is None:
                if self.velocity_max > 0 and self.counter > COUNTER_TRIGGER: # going forward
                    self.get_logger().info(f'Joint {joint} is not moving, record position')
                    self.joints_dictionary['upper_limit'][idx] = self.joints_dictionary['joint_pos'][idx]
                    self.get_logger().info(f'Upper Limit: {self.joints_dictionary["upper_limit"][idx]}')
                    self.velocity_max = -self.velocity_max
                    self.counter = 0

                if (self.velocity_max < 0 and self.counter > COUNTER_TRIGGER): # going backward
                    self.get_logger().info(f'Joint {joint} is not moving, record position')
                    self.joints_dictionary['lower_limit'][idx] = self.joints_dictionary['joint_pos'][idx]
                    self.get_logger().info(f'Lower Limit: {self.joints_dictionary["lower_limit"][idx]}')
                    self.velocity_max = -self.velocity_max
                    # set center position
                    self.joints_dictionary['center_pos'][idx] = (self.joints_dictionary['upper_limit'][idx] + self.joints_dictionary['lower_limit'][idx])/2
                    self.get_logger().info(f'Center Position: {self.joints_dictionary["center_pos"][idx]}')
                    self.counter = 0

            # center pos is determined, move the joint to the center pos
            if self.joints_dictionary['center_pos'][idx] is not None:
                if np.abs(self.joints_dictionary['joint_pos'][idx] - self.joints_dictionary['center_pos'][idx]) > EPSILON:
                    # drive the robot to the center position by ramping down position
                    self.setpt_pos = self.velocity_max*self.counter_ramp_center*TIME_PERIOD + self.joints_dictionary["lower_limit"][idx]
                    self.setpt_vel = self.velocity_max
                    self.counter_ramp_center += 1

                # verify the center position was achieved
                if np.abs(self.joints_dictionary['joint_pos'][idx] - self.joints_dictionary['center_pos'][idx]) < EPSILON:
                    self.joints_dictionary['is_calibrated'][idx] = True
                    self.velocity_max = -self.velocity_max
                    self.counter = 0
                    self.counter_ramp_center = 0
                    self.get_logger().info(f'Joint {joint} is calibrated')
                    self.get_logger().info(f'Center Position achieved: {self.joints_dictionary["joint_pos"][idx]}')

                # check for overshooting
                if self.velocity_max > 0 and self.joints_dictionary['joint_pos'][idx] > self.joints_dictionary['center_pos'][idx]:
                    self.get_logger().info(f'Overshooting, stop the joint')
                    self.setpt_vel = 0
                    self.joints_dictionary['is_calibrated'][idx] = True
                    self.get_logger().info(f'Joint position achieved: {self.joints_dictionary["joint_pos"][idx]}')
                    self.counter = 0
                    self.counter_ramp_center = 0

                if self.velocity_max < 0 and self.joints_dictionary['joint_pos'][idx] < self.joints_dictionary['center_pos'][idx]:
                    self.get_logger().info(f'Overshooting, stop the joint')
                    self.get_logger().info(f'Joint position achieved: {self.joints_dictionary["joint_pos"][idx]}')
                    self.setpt_vel = 0
                    self.joints_dictionary['is_calibrated'][idx] = True
                    self.counter = 0
                    self.counter_ramp_center = 0

            # center pos was not determined, move the joint
            if self.joints_dictionary['center_pos'][idx] is None:
                self.setpt_pos = self.velocity_max*self.counter*(TIME_PERIOD*0.1) + self.joints_dictionary['initial_pos'][idx]
                self.setpt_vel = self.velocity_max


    def timer_callback(self):
        if self.joints_dictionary['is_calibrated'] == [True]*len(self.joints_dictionary['joint_names']):
            self.all_motors_calibrated = True
            # print once
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