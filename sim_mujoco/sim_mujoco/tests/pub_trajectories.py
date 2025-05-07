import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist

class MultiDOFTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('multi_dof_trajectory_publisher')
        self.publisher_ = self.create_publisher(MultiDOFJointTrajectory, '/body_trajectories', 10)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        trajectory_msg = MultiDOFJointTrajectory()

        trajectory_msg.joint_names.append("L_ANKLE")
        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = 0.1
        transform.translation.y = 0.0
        transform.translation.z = 0.2
        transform.rotation.x = 0.0
        transform.rotation.y = 0.0
        transform.rotation.z = 0.0
        transform.rotation.w = 1.0

        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        acc = Twist()
        acc.linear.x = 0.0
        acc.linear.y = 0.0
        acc.linear.z = 0.0
        acc.angular.x = 0.0
        acc.angular.y = 0.0
        acc.angular.z = 0.0

        point.transforms.append(transform)
        point.velocities.append(twist)
        point.accelerations.append(acc)

        point.time_from_start.sec = 0
        trajectory_msg.points.append(point)

        self.publisher_.publish(trajectory_msg)
        self.get_logger().info('Publishing MultiDOFJointTrajectory')

def main(args=None):
    rclpy.init(args=args)
    multi_dof_trajectory_publisher = MultiDOFTrajectoryPublisher()
    rclpy.spin(multi_dof_trajectory_publisher)

    multi_dof_trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
