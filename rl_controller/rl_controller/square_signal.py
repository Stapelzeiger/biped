import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')

        # joint_states
        self.subscription = self.create_subscription(
            JointState,
            '~/joint_states',
            self.joint_states_cb,
            10)
        self.subscription  # prevent unused variable warning
        # Lock.
        self.lock = threading.Lock()

        # publish joint_trajectory
        self.publisher_joints = self.create_publisher(JointTrajectory, '~/joint_trajectory', 10)


    def joint_states_cb(self, msg: JointState):
        # as we get the message,
        with self.lock:
            self.joints_msg = msg

            # process the message and convert to joint trajectory
            # TODO: convert this into a joint Trajectry and then publush.

            # msg = String()
            # msg.data = 'Hello World'
            # self.publisher_.publish(msg)
            # self.get_logger().info('Publishing: "%s"' % msg.data)
            msg = JointTrajectory()
            



def main(args=None):
    try:
        with rclpy.init(args=args):
            minimal_publisher = MinimalPublisher()

            rclpy.spin(minimal_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()