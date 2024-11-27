import rclpy
from rclpy.node import Node

from biped_bringup.msg import StampedBool

import numpy as np

class PubBaselinkContact(Node):

    def __init__(self):
        super().__init__('baselink_contact')
        self.publisher_ = self.create_publisher(StampedBool, '/ik_interface/contact_base_link', 10)
        self.timer_period = 0.01  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.t = 0

    def timer_callback(self):
        msg = StampedBool()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.data = True
        self.publisher_.publish(msg)
        self.t += self.timer_period


def main(args=None):
    rclpy.init(args=args)
    pub_baselink_contact = PubBaselinkContact()
    rclpy.spin(pub_baselink_contact)
    pub_baselink_contact.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()