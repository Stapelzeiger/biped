import rclpy
from rclpy.node import Node
import threading
from std_msgs.msg import Bool, Float64, Empty
from sensor_msgs.msg import Joy
import numpy as np

from geometry_msgs.msg import TwistStamped

class JoyNode(Node):

    def __init__(self):
        super().__init__('joy_command')
        self.lock = threading.Lock()
        self.twist_pub = self.create_publisher(TwistStamped, "~/commanded_twist", 1)
        self.pub_step_sim = self.create_publisher(Float64, "~/step_sim", 1)
        self.pub_pause_sim = self.create_publisher(Bool, "~/pause_sim", 1)
        self.pub_reset_sim = self.create_publisher(Empty, "~/reset_sim", 1)
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_cb,
            1)
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.timer_cb)
        self.joy = None

        self.pause_btn = False
        self.paused = True
        self.step_btn = False
        self.speed_step_cnt = 0
        self.counter = 0.0

        self.list_desired_vel = [
            np.array([0.0, 0.0]),
            np.array([0.02, 0.0]),
            np.array([0.03, 0.0]),
        ]
        print(self.list_desired_vel)
        self.current_des_vel = self.list_desired_vel[0]
        self.use_joystick = False
        if self.use_joystick == False:
            msg = Empty()
            self.pub_reset_sim.publish(msg)

    def joy_cb(self, msg):
        with self.lock:
            self.joy = msg

    def timer_cb(self):
        with self.lock:
            joy = self.joy
        if joy is None:
            return

        if self.use_joystick == False:
            duration_traj = 40.0 # should be higher than 15, cause initialization takes some time
            N = duration_traj/self.dt
            print(self.counter)
            if self.counter % int(N) == 0:
                self.list_desired_vel.pop(0)
                self.counter = 0
                msg = Empty()
                self.pub_reset_sim.publish(msg)
                print('moving to a new trajectory: ', self.current_des_vel)

            if self.counter < N/2:
                self.current_des_vel = np.array([0.0, 0.0])
            else:
                self.current_des_vel = self.list_desired_vel[0]

            msg_twist = TwistStamped()
            msg_twist.header.stamp = self.get_clock().now().to_msg()
            msg_twist.header.frame_id = "base_link"
            msg_twist.twist.linear.x = self.current_des_vel[0]
            msg_twist.twist.linear.y = self.current_des_vel[1]
            msg_twist.twist.linear.z = 0.0
            msg_twist.twist.angular.x = 0.0
            msg_twist.twist.angular.y = 0.0
            msg_twist.twist.angular.z = 0.0
            self.twist_pub.publish(msg_twist)
        else:
            msg_twist = TwistStamped()
            msg_twist.header.stamp = self.get_clock().now().to_msg()
            msg_twist.header.frame_id = "base_link"
            msg_twist.twist.linear.x = 0.1 * joy.axes[1]
            msg_twist.twist.linear.y = 0.1 * joy.axes[0]
            msg_twist.twist.linear.z = 0.0
            msg_twist.twist.angular.x = 0.0
            msg_twist.twist.angular.y = 0.0
            msg_twist.twist.angular.z = 0.1 * joy.axes[3]
            self.twist_pub.publish(msg_twist)


        if self.joy.buttons[0] and not self.pause_btn: # A
            self.paused = not self.paused
        self.pause_btn = joy.buttons[0]

        msg_pause_msg = Bool()
        msg_pause_msg.data = self.paused
        self.pub_pause_sim.publish(msg_pause_msg)

        speed_step = (1 - joy.axes[5]) / 2 # R2
        if speed_step > 0.02:
            self.speed_step_cnt += speed_step
            while self.speed_step_cnt > 1:
                self.speed_step_cnt -= 1
                msg = Float64()
                msg.data = self.dt
                self.pub_step_sim.publish(msg)

        if self.joy.buttons[5] and not self.step_btn: # R1
            msg = Float64()
            msg.data = 0.000001
            self.pub_step_sim.publish(msg)
        self.step_btn = joy.buttons[5]

        if self.joy.buttons[2]: # X
            msg = Empty()
            self.pub_reset_sim.publish(msg)
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = JoyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()