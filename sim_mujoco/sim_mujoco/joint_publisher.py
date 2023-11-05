import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

class JointTrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('joint_publisher')
        self.publisher_ = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.timer_period = 0.01  # seconds
        self.t = 0.0
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        msg = JointTrajectory()
        msg.joint_names = ["R_KFE", "L_KFE"]
        r_kfe_traj = np.sin(2*np.pi*self.t)
        l_kfe_traj = np.cos(2*np.pi*self.t)
        r_kfe_vel = 2*np.pi*np.cos(2*np.pi*self.t)
        l_kfe_vel = -2*np.pi*np.sin(2*np.pi*self.t)
        point = JointTrajectoryPoint()
        point.positions = [r_kfe_traj, l_kfe_traj]
        point.velocities = [r_kfe_vel, l_kfe_vel]
        msg.points.append(point)
        self.publisher_.publish(msg)
        self.t += self.timer_period

def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointTrajectoryPublisher()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()