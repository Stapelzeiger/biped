"""
This file is used to see the stepping stones in the rviz simulation environment.

MuJoCo already simulates it in custom_robot.mujoco.xml file. But rviz can't see that
So, we need to add markers in the rviz environment to see the stepping stones.
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import ColorRGBA

class BoxMarkerPublisher(Node):
    def __init__(self):
        super().__init__('box_marker_publisher')
        self.publisher_ = self.create_publisher(Marker, 'visualization_marker', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.box_marker = Marker()
        # TODO: fix so that it actually puts it in the right position, the frame matters here
        self.box_marker.header.frame_id = "odom"
        self.box_marker.ns = "box"
        self.box_marker.id = 0
        self.box_marker.type = Marker.CUBE
        self.box_marker.action = Marker.ADD
        self.box_marker.pose.position.x = 0.2
        self.box_marker.pose.position.y = 0.15
        self.box_marker.pose.position.z = 0.0
        self.box_marker.pose.orientation.w = 0.0
        self.box_marker.scale.x = 0.1
        self.box_marker.scale.y = 0.1
        self.box_marker.scale.z = 0.05
        self.box_marker.color.a = 1.0
        self.box_marker.color.r = 1.0
        self.box_marker.color.g = 0.0
        self.box_marker.color.b = 0.0

    def timer_callback(self):
        self.publisher_.publish(self.box_marker)

def main(args=None):
    rclpy.init(args=args)

    box_marker_publisher = BoxMarkerPublisher()

    rclpy.spin(box_marker_publisher)

    box_marker_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
