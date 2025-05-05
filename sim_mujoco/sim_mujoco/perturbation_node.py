#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import numpy as np

class PerturbationNode(Node):
    def __init__(self):
        super().__init__('perturbation_node')

        self.publisher_ = self.create_publisher(WrenchStamped, '/wrench_perturbation', 10)

        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.force_amplitude = 10.0  # N
        self.torque_amplitude = 5.0  # Nm

    def timer_callback(self):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        force_direction = np.random.randn(3)
        force_direction = force_direction / np.linalg.norm(force_direction)
        force = force_direction * self.force_amplitude

        torque_direction = np.random.randn(3)
        torque_direction = torque_direction / np.linalg.norm(torque_direction)
        torque = torque_direction * self.torque_amplitude

        msg.wrench.force.x = force[0]
        msg.wrench.force.y = force[1]
        msg.wrench.force.z = force[2]
        msg.wrench.torque.x = torque[0]
        msg.wrench.torque.y = torque[1]
        msg.wrench.torque.z = torque[2]

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    perturbation_node = PerturbationNode()
    rclpy.spin(perturbation_node)
    perturbation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
