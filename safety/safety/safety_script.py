import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosgraph_msgs.msg import Clock


class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_script')
        self.joint_traj_pub = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)

        self.dt = 0.02
        self.time = 0
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.timer = self.create_timer(self.dt, self.pub_safety_traj)
        self.declare_parameter('joint_names')
        self.joint_names = self.get_parameter('joint_names') # todo, this doesn't work yet

        self.joint_names = ['FR_YAW', 'FR_HAA', 'FR_HFE', 'FR_KFE', 'FL_YAW', 'FL_HAA', 'FL_HFE', 'FL_KFE']

        print(self.joint_names)
        
    def pub_safety_traj(self):

        clock_msg = Clock()
        clock_msg.clock.sec = int(self.time)
        clock_msg.clock.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)
        self.clock_pub.publish(clock_msg)

        msg = JointTrajectory()
        msg.header.stamp.sec = int(self.time)
        msg.header.stamp.nanosec = int((self.time - clock_msg.clock.sec) * 1e9)

        msg = JointTrajectory()
        msg.joint_names = self.joint_names


        for idx, joint in enumerate(self.joint_names):
            msg_point = JointTrajectoryPoint()
            msg_point.positions.append(0.0)
            msg_point.velocities.append(0.0)
            msg.points.append(msg_point)

        self.joint_traj_pub.publish(msg)

        self.time += self.dt


def main(args=None):
    rclpy.init(args=args)
    sim_node = SafetyNode()
    rclpy.spin(sim_node)
    sim_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
