import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from threading import Lock
import math

def yaw_from_quaternion(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

def wrap_angle(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi

def clamp(x, min_x, max_x):
    return min(max_x, max(min_x, x))

class PosCtrl(Node):

    def __init__(self):
        super().__init__('pos_ctrl')
        self.lock = Lock()
        self.goal = None
        self.vel_pub = self.create_publisher(TwistStamped, '~/cmd_vel', 10)
        self.goal_sub = self.create_subscription(PoseStamped, '~/goal', self.goal_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, '~/odometry', self.odom_callback, 1)
        self.vx_max = 0.1
        self.vy_max = 0.05
        self.omega_max = 0.05
        self.kpos = 0.1
        self.kyaw = 0.1

    def goal_callback(self, msg):
        with self.lock:
            self.goal = msg

    def odom_callback(self, msg):
        with self.lock:
            if self.goal is None:
                return
            if self.goal.header.frame_id != msg.header.frame_id:
                self.get_logger().warn('Goal and odometry frames do not match')
                return
            goal_pos = self.goal.pose.position
            goal_yaw = yaw_from_quaternion(self.goal.pose.orientation)
            robot_pos = msg.pose.pose.position
            robot_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
            dx = goal_pos.x - robot_pos.x
            dy = goal_pos.y - robot_pos.y
            dyaw = wrap_angle(goal_yaw - robot_yaw)
            vx = clamp(self.kpos*dx, -self.vx_max, self.vx_max)
            vy = clamp(self.kpos*dy, -self.vy_max, self.vy_max)
            omega = clamp(self.kyaw*dyaw, -self.omega_max, self.omega_max)
            twist = TwistStamped()
            twist.header.stamp = msg.header.stamp
            twist.header.frame_id = msg.header.frame_id
            twist.twist.linear.x = vx
            twist.twist.linear.y = vy
            twist.twist.angular.z = omega
            self.vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    pos_ctrl = PosCtrl()
    rclpy.spin(pos_ctrl)
    pos_ctrl.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()