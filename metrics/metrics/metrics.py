import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

class MetricsPublisher(Node):

    def __init__(self):
        super().__init__('metrics_publisher')
        self.publisher_speed = self.create_publisher(Float32, '~/speed', 10)
        self.subscriber_odom = self.create_subscription(Odometry, 'odometry', self.odom_callback, 10)

    def odom_callback(self, msg):
        robot_speed = msg.twist.twist.linear.x
        msg_speed = Float32()
        msg_speed.data = robot_speed
        self.publisher_speed.publish(msg_speed)

def main(args=None):
    rclpy.init(args=args)
    metrics_publisher = MetricsPublisher()
    rclpy.spin(metrics_publisher)
    metrics_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()