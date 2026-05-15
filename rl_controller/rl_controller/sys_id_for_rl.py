from trajectory_msgs.msg import JointTrajectory
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
import os
from etils import epath
import json
import threading

from scipy.spatial.transform import Rotation as R


import xml.etree.ElementTree as ET
from std_msgs.msg import String
from sensor_msgs.msg import JointState

from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter

import shutil


def get_square_signal_value(current_time_sec, period, min_limit, max_limit):
    """
    Generates a square wave alternating between min_limit and max_limit.
    """
    min_lim = float(min_limit)
    max_lim = float(max_limit)
    
    if period <= 0:
        return max_lim # Fallback to avoid division by zero
        
    # Determine where we are in the current cycle
    phase = current_time_sec % period
    
    # First half of the period is max, second half is min
    if phase < (period / 2.0):
        return max_lim
    else:
        return min_lim


class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher_rl')

        self.dt_ctrl = 0.01

        # Read the URDF file for the robot to ensure we have the correct joint names.
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.urdf_sub = self.create_subscription(
            String,
            '/robot_description',
            self.urdf_callback,
            qos_profile  # Apply QoS profile
        )
        self.urdf_sub = None
        self.joints_from_urdf = {}

        # Lock.
        self.lock = threading.Lock()

        self.joints_msg = None
        self.joint_states_sub = self.create_subscription(
            JointState,
            '~/joint_states',
            self.joint_states_cb,
            1)

        # TODO: put _compensated in the name of the topic for hardware.
        self.publisher_joints = self.create_publisher(JointTrajectory, '~/joint_trajectory', 10)
        self.timer = self.create_timer(self.dt_ctrl, self.step_controller)


    def urdf_callback(self, msg: String):
        """Extracts all joint names along with their min and max limits from the URDF."""
        urdf_str = msg.data
        try:
            root = ET.fromstring(urdf_str)
            for joint in root.findall('joint'):
                name = joint.get('name')
                limit = joint.find('limit')
                min_limit = limit.get('lower') if limit is not None else "N/A"
                max_limit = limit.get('upper') if limit is not None else "N/A"
                if min_limit == "N/A" or max_limit == "N/A":
                    self.get_logger().warn(f"Joint {name} has no limits. Will not add to the dictionary.")
                else:
                    self.joints_from_urdf[name] = (min_limit, max_limit)
        except ET.ParseError as e:
            self.get_logger().error(f"Failed to parse URDF: {e}")
        self.get_logger().info(f'Extracted Joints: {self.joints_from_urdf}')
        # assert len(self.joints_from_urdf.keys()) == self.action_size, \
        #     f"Number of joints in URDF ({len(self.joints_from_urdf.keys())}) does not match the action size ({self.action_size})."


    def joint_states_cb(self, msg: JointState):
        with self.lock:
            self.joints_msg = msg

    def step_controller(self):
        time_now = self.get_clock().now().nanoseconds / 1e9
        if self.joints_from_urdf == {}:
            self.get_logger().error('Joint limits not set. Cannot publish trajectory.')
            return

        if self.joints_msg is None:
            self.get_logger().warn('Joint data not received. Skipping this step.')
            return

        joints_state_time = self.joints_msg.header.stamp.sec + self.joints_msg.header.stamp.nanosec / 1e9
        if abs(time_now - joints_state_time) > 0.1:
            self.get_logger().warn('Joint state data is old. Skipping this step.')
            return

        joint_names = ['R_HAA']
        self.publish_joints(joint_names)


    def publish_joints(self, joint_names: list):
        new_msg = JointTrajectory()
        # process the message and convert to joint trajectory
        # TODO: convert this into a joint Trajectry and then publush.
        new_msg.joint_names = joint_names
        new_msg.header.stamp = self.get_clock().now().to_msg()

        # Get current time in seconds for the signal wave
        current_time_sec = self.get_clock().now().nanoseconds / 1e9
        
        # Define your period parameter (e.g., 2.0 seconds per full cycle)
        signal_period = 2

        point = JointTrajectoryPoint()
        joints_out = []
        for joint_name in new_msg.joint_names:
            # self.get_logger().info(f"Processing joint: {joint_name}")
            (min_limit, max_limit) = self.joints_from_urdf[joint_name]
            fifty_percent_min_limit = 50/100 * float(min_limit)
            fifty_percent_max_limit = 50/100 * float(max_limit)

            # Write square signal code
            # depending if 1 or 0, then we will publish the max or min limit.
            value = get_square_signal_value(current_time_sec, signal_period, fifty_percent_min_limit, fifty_percent_max_limit)
            joints_out.append(value)
        
        point.positions = joints_out
        point.velocities = [0.0] * len(joints_out)
        point.effort = [0.0] * len(joints_out)
        new_msg.points.append(point)
        self.publisher_joints.publish(new_msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()