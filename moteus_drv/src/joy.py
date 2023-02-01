import rclpy
from rclpy.node import Node
import threading

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Joy

class JoyNode(Node):
    def __init__(self):
        super().__init__('JoyMotorControl')
        self.lock = threading.Lock()
        self.pub = self.create_publisher(JointTrajectory, "/joint_trajectory", 1)
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_cb,
            1)
        self.timer = self.create_timer(0.01, self.timer_cb)
        self.joy = None

    def timer_cb(self):
        with self.lock:
            joy = self.joy
        if joy is None:
            return
        ctrl = JointTrajectory()
        ctrl.header.stamp = self.get_clock().now().to_msg()
        ctrl.joint_names = ['R_YAW']
        ctrl.points = [JointTrajectoryPoint()]
        ctrl.points[0].positions.append(joy.axes[5]*3.15*0.5*0.2)
        ctrl.points[0].velocities.append(0)
        ctrl.points[0].effort.append(0)
        self.pub.publish(ctrl)

    def joy_cb(self, msg):
        with self.lock:
            self.joy = msg



def main(args=None):
    rclpy.init(args=args)
    node = JoyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()