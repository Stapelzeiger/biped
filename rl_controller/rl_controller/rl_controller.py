from trajectory_msgs.msg import JointTrajectory
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
from jax import numpy as jp
import os
from etils import epath
import sys

jax.config.update('jax_platform_name', 'cpu')

# Load policy.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
print("Available devices:", jax.devices())
print(str(jax.local_devices()[0]))

RESULTS_FOLDER_PATH ='/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/rl_controller/results'

class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher_rl')
        self.publisher_ = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.timer = self.create_timer(0.001, self.publish_trajectory)  # Publish every second
        self.joint_names = ['L_YAW', 'L_HAA', 'L_HFE', 'L_KFE',
                            'R_YAW', 'R_HAA', 'R_HFE', 'R_KFE',] # TODO: don't make this hardcoded
        self.counter = 0

        # Load RL policy.
        # Sort by date and get the latest folder.
        folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
        latest_folder = folders[-1]
        folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
        folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]
        if len(folders) == 0:
            raise ValueError(f'No folders found in {epath.Path(RESULTS_FOLDER_PATH) / latest_folder}')
            sys.exit()
        if len(folders) > 1:
            latest_weights_folder = folders[-1]
        else:
            latest_weights_folder = folders
        print(f'Latest weights folder: {latest_weights_folder}')

        path = epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder
        print(f'Loading policy from: {path}')

        self.policy_fn = ppo_checkpoint.load_policy(path)
        self.jit_policy = jax.jit(self.policy_fn)
        self.rng = jax.random.PRNGKey(1)

    def generate_trajectory_point(self, t):
        point = JointTrajectoryPoint()
        obs = {
            'privileged_state': jp.zeros(102),
            'state': jp.zeros(46),
            }
        # obs['state'] = np.array(

        # )

        act_rng, self.rng = jax.random.split(self.rng)
        ctrl, _ = self.jit_policy(obs, act_rng)

        point.positions = ctrl.tolist()
        point.velocities = (0.1 * np.cos(0.1 * t + np.arange(len(self.joint_names)))).tolist()
        point.accelerations = (-0.01 * np.sin(0.1 * t + np.arange(len(self.joint_names)))).tolist()
        point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        return point

    def publish_trajectory(self):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_point = self.generate_trajectory_point(self.counter)
        msg.points.append(trajectory_point)
        self.publisher_.publish(msg)
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()