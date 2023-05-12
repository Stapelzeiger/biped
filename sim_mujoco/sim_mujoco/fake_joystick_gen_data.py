import rclpy
from rclpy.node import Node
import threading
from std_msgs.msg import Bool, Float64, Empty
from sensor_msgs.msg import Joy
import numpy as np

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

class JoyNode(Node):

    def __init__(self):
        super().__init__('joy_command')
        self.lock = threading.Lock()
        self.twist_pub = self.create_publisher(TwistStamped, "~/commanded_twist", 1)
        self.pub_step_sim = self.create_publisher(Float64, "~/step_sim", 1)
        self.pub_pause_sim = self.create_publisher(Bool, "~/pause_sim", 1)
        self.pub_reset_sim = self.create_publisher(Empty, "~/reset_sim", 1)

        self.create_subscription(Odometry, "/odometry", self.odom_cb, 1)

        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.timer_cb)

        self.paused = False
        self.counter_time = 0
        self.desired_forward_speed = 0.3
        self.desired_lateral_speed = 0.0
        self.desired_angular_speed = 0.0
        self.sequence_duration = 20
        self.pos_x_I = 0.0
        self.pos_y_I = 0.0
        self.pos_z_I = 0.0


    def odom_cb(self, msg):
        self.pos_x_I = msg.pose.pose.position.x
        self.pos_y_I = msg.pose.pose.position.y
        self.pos_z_I = msg.pose.pose.position.z

    def timer_cb(self):

        msg_pause_msg = Bool()
        msg_pause_msg.data = self.paused
        self.pub_pause_sim.publish(msg_pause_msg)

        msg_twist = TwistStamped()
        msg_twist.header.stamp = self.get_clock().now().to_msg()
        msg_twist.header.frame_id = "base_link"
        msg_twist.twist.linear.x = 0.1*self.desired_forward_speed
        msg_twist.twist.linear.y = 0.1*self.desired_lateral_speed
        msg_twist.twist.linear.z = 0.0
        msg_twist.twist.angular.x = 0.0
        msg_twist.twist.angular.y = 0.0
        msg_twist.twist.angular.z = 0.1*self.desired_angular_speed
        self.twist_pub.publish(msg_twist)

        if self.counter_time > self.sequence_duration and self.counter_time < self.sequence_duration + 10:
            print('reset')
            msg = Empty()
            self.pub_reset_sim.publish(msg)
            self.counter_time = 0
            self.desired_forward_speed = min(self.desired_forward_speed + 0.1, 1)
            self.desired_lateral_speed = 0.0
            self.desired_angular_speed = 0.0
            print(self.desired_forward_speed)

        self.counter_time = self.counter_time + self.dt


def main(args=None):
    rclpy.init(args=args)
    node = JoyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()