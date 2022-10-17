import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock

import mujoco

class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.time = 0
        self.dt = 0.1
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.timer = self.create_timer(self.dt, self.step)

    def step(self):
        self.time += self.dt

        msg = JointState()
        self.joint_state_pub.publish(msg)

        clock_msg = Clock()
        clock_msg.clock.sec = int(self.time)
        clock_msg.clock.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        self.clock_pub.publish(clock_msg)

def main(args=None):
    rclpy.init(args=args)
    sim_node = MujocoNode()
    rclpy.spin(sim_node)
    sim_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
