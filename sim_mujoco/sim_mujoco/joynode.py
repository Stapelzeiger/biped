import rclpy
from rclpy.node import Node
import threading

from sensor_msgs.msg import Joy
import numpy as np

from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory
from geometry_msgs.msg import Transform

class JoyNode(Node):
    def __init__(self):
        super().__init__('JoyFootDesiredPos')
        self.lock = threading.Lock()
        self.foot_position_BL_pub = self.create_publisher(MultiDOFJointTrajectory, "/body_trajectories", 1)
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_cb,
            1)
        self.timer = self.create_timer(0.005, self.timer_cb)
        self.joy = None

    def joy_cb(self, msg):
        with self.lock:
            self.joy = msg
    
    def timer_cb(self):
        with self.lock:
            joy = self.joy
        if joy is None:
            return

        input_joystick = [0.1*joy.axes[1], 0.1*joy.axes[0], 0.1*joy.axes[4]]
        yaw_angle_foot = 0.5*joy.axes[3]

        msg_foot = MultiDOFJointTrajectory()
        msg_foot.header.stamp = self.get_clock().now().to_msg()
        msg_foot.header.frame_id = "base_link"
        foot_pos_point = MultiDOFJointTrajectoryPoint()

        transforms = Transform()
        transforms.translation.x = input_joystick[0]
        transforms.translation.z = -0.5 + input_joystick[2]
        transforms.rotation.x = 0.0
        transforms.rotation.y = 0.0
        transforms.rotation.z = np.sin(yaw_angle_foot/2)
        transforms.rotation.w = np.cos(yaw_angle_foot/2)
        transforms.translation.y = -0.1 + input_joystick[1]
        msg_foot.joint_names.append("R_ANKLE")
        foot_pos_point.transforms.append(transforms)

        transforms = Transform()
        transforms.translation.x = input_joystick[0]
        transforms.translation.z = -0.5 + input_joystick[2]
        transforms.rotation.x = 0.0
        transforms.rotation.y = 0.0
        transforms.rotation.z = np.sin(yaw_angle_foot/2)
        transforms.rotation.w = np.cos(yaw_angle_foot/2)
        transforms.translation.y = 0.1 + input_joystick[1]
        msg_foot.joint_names.append("L_ANKLE")
        foot_pos_point.transforms.append(transforms)

        msg_foot.points = [foot_pos_point]
        self.foot_position_BL_pub.publish(msg_foot)

        print('des foot loc:', transforms.translation.x, transforms.translation.y, transforms.translation.z)

def main(args=None):
    rclpy.init(args=args)
    node = JoyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()