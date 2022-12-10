import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from rosgraph_msgs.msg import Clock


class SafetyNode(Node):
    def __init__(self):
        super().__init__('foot_sensor')
        self.foot_sensor_right_pub = self.create_publisher(Bool, '/foot_sensor_right', 10)
        self.foot_sensor_left_pub = self.create_publisher(Bool, '/foot_sensor_left', 10)

        self.dt = 0.02
        self.time = 0
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.timer = self.create_timer(self.dt, self.pub_foot_sensors)

        
    def pub_foot_sensors(self):

        clock_msg = Clock()
        clock_msg.clock.sec = int(self.time)
        clock_msg.clock.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        self.clock_pub.publish(clock_msg)

        msg_foot_sensor_right = Bool()
        msg_foot_sensor_right.data = True

        msg_foot_sensor_left = Bool()
        msg_foot_sensor_left.data = False
        
        self.foot_sensor_right_pub.publish(msg_foot_sensor_right)
        self.foot_sensor_left_pub.publish(msg_foot_sensor_left)

        self.time += self.dt


def main(args=None):
    rclpy.init(args=args)
    sim_node = SafetyNode()
    rclpy.spin(sim_node)
    sim_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
