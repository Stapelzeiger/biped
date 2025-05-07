#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Quaternion
import threading

class PerturbationNode(Node):
    def __init__(self):
        super().__init__('perturbation_node')

        self.publisher_ = self.create_publisher(WrenchStamped, '/wrench_perturbation_I', 10)
        self.odom_robot = self.create_subscription(Odometry, '/odometry', self.odom_cb, 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.robot_position = None
        self.robot_orientation = None

        self.force_amplitude = 15.0  # N
        self.torque_amplitude = 0.6  # Nm

        # Duty cycle parameters
        self.cycle_period = 1.0
        self.duty_cycle = 0.1
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        self.force = None
        self.torque = None

        self.lock = threading.Lock()

        timer_period = 0.01  # seconds - higher frequency for smoother duty cycle
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def odom_cb(self, msg):
        # Store robot position and orientation
        with self.lock:
            self.robot_pos_I = msg.pose.pose.position

            # Publish TF
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = 'robot_centric_world'

            # Set translation to robot's position
            t.transform.translation.x = self.robot_pos_I.x
            t.transform.translation.y = self.robot_pos_I.y
            t.transform.translation.z = 1.0

            # Set rotation to identity (aligned with world frame)
            t.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            self.tf_broadcaster.sendTransform(t)

    def timer_callback(self):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'robot_centric_world'

        current_time = self.get_clock().now().nanoseconds / 1e9
        cycle_time = (current_time - self.start_time) % self.cycle_period

        # Check if we're in the "on" portion of the duty cycle
        if cycle_time < (self.cycle_period * self.duty_cycle):
            if self.force is None:
                force_direction = np.random.randn(2)
                force_direction = force_direction / np.linalg.norm(force_direction)
                self.force = force_direction * self.force_amplitude
                self.get_logger().info(f"Force: {self.force}")

            if self.torque is None:
                torque_direction = np.random.randn(3)
                torque_direction = torque_direction / np.linalg.norm(torque_direction)
                self.torque = torque_direction * self.torque_amplitude
                self.get_logger().info(f"Torque: {self.torque}")
            msg.wrench.force.x = self.force[0]
            msg.wrench.force.y = self.force[1]
            msg.wrench.force.z = 0.0
            msg.wrench.torque.x = self.torque[0]
            msg.wrench.torque.y = self.torque[1]
            msg.wrench.torque.z = self.torque[2]

        else:
            self.force = None
            self.torque = None
            msg.wrench.force.x = 0.0
            msg.wrench.force.y = 0.0
            msg.wrench.force.z = 0.0
            msg.wrench.torque.x = 0.0
            msg.wrench.torque.y = 0.0
            msg.wrench.torque.z = 0.0

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    perturbation_node = PerturbationNode()
    rclpy.spin(perturbation_node)
    perturbation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
