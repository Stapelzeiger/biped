import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import numpy as np

class MotorTest(Node):

    def __init__(self):
        super().__init__('motor_test')
        self.publisher_ = self.create_publisher(JointTrajectory, '~/joint_trajectory', 10)
        self.timer_period = 0.01  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.t = 0

    def timer_callback(self):
        f = 4
        A = 0.2
        w = f * 2 * np.pi
        pos = A * np.sin(w*self.t)
        vel = A * w * np.cos(w*self.t)
        acc = -A * w**2 * np.sin(w*self.t)
        m = 0.236
        l = 0.23
        I = 2 * m * l**2
        tau = I * acc
        # tau = 0

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.joint_names = ['test_motor']
        msg.points = [JointTrajectoryPoint(positions=[pos], velocities=[vel], accelerations=[0], effort=[tau])]
        self.publisher_.publish(msg)
        self.t += self.timer_period


def main(args=None):
    rclpy.init(args=args)

    motor_test = MotorTest()

    rclpy.spin(motor_test)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    motor_test.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()