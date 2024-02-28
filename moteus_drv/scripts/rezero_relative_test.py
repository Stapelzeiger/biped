#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np
import threading

TIME_PERIOD = 0.01
VEL_MAX = 0.1
MAX_TRIGGER_EFFORT = 1.5
COUNTER_TRIGGER = 20
EPSILON = 0.01


class JointCalibration(Node):
    def __init__(self):
        super().__init__('joint_calibration_publisher')
        # this will get moteus_drv share folder.
        # on the rasberrypi: /home/biped-raspi/biped_ws/install/moteus_drv/share/moteus_drv
        self.ws_share_folder = self.declare_parameter('install_folder', rclpy.Parameter.Type.STRING).value

        list_motors = self.declare_parameter('joints', rclpy.Parameter.Type.STRING_ARRAY).value
        self.get_logger().info(f'List of motors: {list_motors}')

        self.joints_dictionary = { # TODO populate this dict from the config params.yaml
            'joint_names': list_motors,
            'is_calibrated': [False]*len(list_motors),
            'joint_pos': [None]*len(list_motors),
            'joint_vel': [None]*len(list_motors),
            'joint_effort': [None]*len(list_motors),
            'center_pos': [None]*len(list_motors),
            'upper_limit': [None]*len(list_motors),
            'lower_limit': [None]*len(list_motors),
            'initial_pos': [None]*len(list_motors),
        }

        for i, joint_name in enumerate(self.joints_dictionary['joint_names']):
            offset_param_str = f'{joint_name}/offset'
            self.declare_parameter(offset_param_str, 0.0)


        self.lock = threading.Lock()

        self.counter = 0
        self.counter_ramp_center = 0
        self.write_offsets = False

        self.velocity_max = VEL_MAX
        if self.velocity_max < 0:
            self.get_logger().info('Starting velocity max cannot be negative')
            return

        self.setpt_pos = None
        self.setpt_vel = None

        self.pub_trajectory = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.joint_states_sub = self.create_subscription(JointState, 'joint_states', self.joint_states_callback, 10)
        self.timer_period = TIME_PERIOD # seconds

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

    def get_calibration_setpt_msg(self, joint, idx):
        with self.lock:
            joint_effort = self.joints_dictionary['joint_effort'][idx]
            # TODO: I want it to go to a specific position gotta figure out the trajectory
            # Since once I force it to go a specific trajectory, I can see how different starting positions affect the motor.
            if self.counter < 360:
                self.setpt_pos = self.velocity_max*self.counter*(TIME_PERIOD*0.1) + self.joints_dictionary['initial_pos'][idx]
                self.get_logger().info(f'setpt_pos: {self.setpt_pos}')
                self.setpt_vel = self.velocity_max
            else:
                self.setpt_pos = self.joints_dictionary['initial_pos'][idx]
                self.setpt_vel = 0.0

            joint_traj_pt_msg = JointTrajectoryPoint()
            joint_traj_pt_msg.positions.append(self.setpt_pos)
            joint_traj_pt_msg.velocities.append(self.setpt_vel)
            joint_traj_pt_msg.effort.append(0)

            return joint_traj_pt_msg

    def timer_callback(self):
        if None in self.joints_dictionary['joint_pos'] or \
            None in self.joints_dictionary['joint_vel'] or \
            None in self.joints_dictionary['joint_effort'] or \
            None in self.joints_dictionary['initial_pos']:
            self.get_logger().info('No joint states received yet')
            return

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        # take the first uncalibrated joint and calibrate it
        for i, joint in enumerate(self.joints_dictionary['joint_names']):
            if self.joints_dictionary['is_calibrated'][i] == False:
                # self.get_logger().info(f'Calibrating joint {joint}')
                joint_traj_pt_msg = self.get_calibration_setpt_msg(joint, i)
                msg.joint_names.append(joint)
                msg.points.append(joint_traj_pt_msg)
                break

        self.pub_trajectory.publish(msg)

        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointCalibration()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()