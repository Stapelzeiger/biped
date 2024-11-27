import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from biped_bringup.msg import StampedBool
import threading

class FootPollMarkerNode(Node):
    def __init__(self):
        super().__init__('foot_poll_marker_node')

        # Publisher for visualization markers
        self.marker_pub_R = self.create_publisher(Marker, '/foot_markers_R', 10)
        self.marker_pub_L = self.create_publisher(Marker, '/foot_markers_L', 10)

        self.lock = threading.Lock()

        # Subscribers for left and right foot poll
        self.l_foot_sub = self.create_subscription(
            StampedBool,
            '/poll_L_FOOT/gpio',
            self.l_foot_callback,
            10
        )
        self.r_foot_sub = self.create_subscription(
            StampedBool,
            '/poll_R_FOOT/gpio',
            self.r_foot_callback,
            10
        )

        # State to store the last received values
        self.l_foot_state = False
        self.r_foot_state = False

    def l_foot_callback(self, msg: StampedBool):
        with self.lock:
            self.l_foot_state = msg.data
            self.publish_marker('l_foot_marker', msg.data, x=0.0, y=0.5, z=0.0, time=msg.header.stamp, side='left', id=0)

    def r_foot_callback(self, msg: StampedBool):
        with self.lock:
            self.r_foot_state = msg.data
            self.publish_marker('r_foot_marker', msg.data, x=0.0, y=-0.5, z=0.0, time=msg.header.stamp, side='right', id=1)

    def publish_marker(self, ns, is_true, x, y, z, time, side='left', id=0):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = time
        marker.ns = ns
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set scale
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Set color based on the boolean value
        if is_true:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

        # Publish marker
        if side == 'left':
            self.marker_pub_L.publish(marker)
        else:
            self.marker_pub_R.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = FootPollMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
