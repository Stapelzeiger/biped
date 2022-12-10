import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
import numpy as np

from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory
from geometry_msgs.msg import Transform

class JoyNode(Node):
    def __init__(self):
        super().__init__('JoyFootDesiredPos')
        self.foot_position_BL_pub = self.create_publisher(MultiDOFJointTrajectory, "/foot_position_BL", 1)
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_cb,
            1)

    def joy_cb(self, msg):

        transforms = Transform()
        input_joystick = [0.1*msg.axes[1], 0.1*msg.axes[0], 0.1*msg.axes[4]]
        yaw_angle_foot = 0.5*msg.axes[3]

        leg_choice = msg.buttons[2] # X on the gamepad


        transforms.translation.x = input_joystick[0]
        transforms.translation.z = -0.5 + input_joystick[2]
        transforms.rotation.x = 0.0
        transforms.rotation.y = 0.0
        transforms.rotation.z = np.sin(yaw_angle_foot/2)
        transforms.rotation.w = np.cos(yaw_angle_foot/2)

        if leg_choice == 0:
            transforms.translation.y = -0.1 + input_joystick[1]
            name_stance_foot = "FR_ANKLE"
            print("FR ANKLE")

        if leg_choice == 1:
            transforms.translation.y = 0.1 + input_joystick[1]
            name_stance_foot = "FL_ANKLE"
            print("FL ANKLE")


        foot_pos_point = MultiDOFJointTrajectoryPoint()
        foot_pos_point.transforms = [transforms]

        msg_foot = MultiDOFJointTrajectory()
        msg_foot.joint_names = [name_stance_foot]
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