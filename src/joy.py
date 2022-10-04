import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Joy

class JoyNode(Node):
    def __init__(self):
        super().__init__('JoyMotorControl')
        self.pub = self.create_publisher(JointTrajectory, "/moteus/joint_traj", 1)
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_cb,
            1)

    def joy_cb(self, msg):
        ctrl = JointTrajectory()
        ctrl.joint_names = ['test']
        ctrl.points = [JointTrajectoryPoint()]
        ctrl.points[0].positions.append(msg.axes[5]*0.1)
        ctrl.points[0].velocities.append(0)
        ctrl.points[0].effort.append(0)
        print(ctrl)
        self.pub.publish(ctrl)
        


def main(args=None):
    rclpy.init(args=args)
    node = JoyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()