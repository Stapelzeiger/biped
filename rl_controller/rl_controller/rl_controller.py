from trajectory_msgs.msg import JointTrajectory
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from biped_bringup.msg import StampedBool

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
from jax import numpy as jp
import os
from etils import epath
import json
import threading

from scipy.spatial.transform import Rotation as R

import xml.etree.ElementTree as ET
from std_msgs.msg import String
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

# Check JAX devices and set the CPU.
jax.config.update('jax_platform_name', 'cpu')
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
print("Available devices:", jax.devices())
print(str(jax.local_devices()[0]))

# TODO: makes these parameters configurable.
RESULTS_FOLDER_PATH ='/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/rl_controller/results'
DT_CTRL = 0.002
TIME_NO_FEET_IN_CONTACT = 0.2

class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher_rl')

        # Load PPO policy.
        self.get_logger().info('Loading PPO policy...')
        latest_results_folder = sorted(os.listdir(RESULTS_FOLDER_PATH))[-1]
        self.get_logger().info(f'    Latest results folder: {latest_results_folder}')
        folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_results_folder))
        folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_results_folder / f)]
        if len(folders) == 0:
            raise ValueError(f'No folders found in {epath.Path(RESULTS_FOLDER_PATH) / latest_results_folder}')
        if len(folders) > 1:
            latest_weights_folder = folders[-1]
        else:
            latest_weights_folder = folders
        self.get_logger().info(f'    Latest weights folder: {latest_weights_folder}')
        path = epath.Path(RESULTS_FOLDER_PATH) / latest_results_folder / latest_weights_folder
        self.get_logger().info(f'    Loading policy from: {path}')
        self.policy_fn = ppo_checkpoint.load_policy(path)
        self.jit_policy = jax.jit(self.policy_fn)
        self.rng = jax.random.PRNGKey(1)

        # Load params of the PPO policy.
        config_file_path = epath.Path(RESULTS_FOLDER_PATH) / latest_results_folder / 'ppo_network_config.json'
        with open(config_file_path) as f:
            self.config = json.load(f)
        self.get_logger().info(f'    Action size: {self.config["action_size"]}, Observation size: {self.config["observation_size"]}')
        self.action_size = self.config['action_size']
        self.obs = {
            'privileged_state': jp.zeros(self.config['observation_size']['privileged_state']),
            'state': jp.zeros(self.config['observation_size']['state'])
        }
        self.joint_names_PPO = ['L_YAW', 'L_HAA', 'L_HFE', 'L_KFE', 'L_ANKLE', \
                                'R_YAW', 'R_HAA', 'R_HFE', 'R_KFE', 'R_ANKLE'] # TODO: don't make this hardcoded, save it in a config file.

        # Read the URDF file for the robot to ensure we have the correct joint names.
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.urdf_sub = self.create_subscription(
            String,
            '/robot_description',
            self.urdf_callback,
            qos_profile  # Apply QoS profile
        )
        self.urdf_sub = None
        self.joints_from_urdf = {}

        # State machine.
        self.state = "INIT"
        self.initialization_done = False

        # Lock.
        self.lock = threading.Lock()

        # Subscribers.
        self.foot_left_contact = False
        self.foot_right_contact = False
        self.foot_left_contact_sub = self.create_subscription(
            StampedBool,
            '~/contact_foot_left',
            self.foot_left_contact_callback,
            1)
        self.foot_right_contact_sub = self.create_subscription(
            StampedBool,
            '~/contact_foot_right',
            self.foot_right_contact_callback,
            1)

        # Subscribers and inputs into the PPO policy.
        self.imu_msg = None
        self.imu_sub = self.create_subscription(
            Imu,
            '~/imu',
            self.imu_cb,
            1)

        self.joints_msg = None
        self.joint_states_sub = self.create_subscription(
            JointState,
            '~/joint_states',
            self.joint_states_cb,
            1)

        self.odom_msg = None
        self.odometry_sub = self.create_subscription(
            Odometry,
            '~/odometry',
            self.odom_cb,
            1)

        # self.gravity_msg = None
        # self.gravity_sub = self.create_subscription(
        #     Vector3Stamped,
        #     '~/gravity',
        #     self.gravity_cb,
        #     1)

        gait_freq = 1.5
        phase_dt = 2 * np.pi * DT_CTRL * gait_freq
        phase = np.array([0, np.pi])

        self.info = {
            'phase': phase,
            'phase_dt': phase_dt,
        }

        self.last_action = np.zeros(self.action_size)

        # Default joint angles.
        self.default_q_joints = np.array([0.0, 0.0, -0.463, 0.983, -0.350, \
                                          0.0, 0.0, -0.463, 0.983, -0.350])
        # self.start_q_joints = np.array([0.0, -0.03, 0.944, -1.598, 0.654, \
                                        #   0.0, 0.23, 0.534, -0.92, 0.386])
        # self.default_q_joints = np.array([0.0, 0.0, 0.5, -0.92, 0.386, \
                                        # 0.0, 0.0, 0.5, -0.92, 0.386])
        self.start_q_joints = self.default_q_joints.copy()
        self.timeout_for_no_feet_in_contact = 0.0

        self.publisher_joints = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.timer = self.create_timer(DT_CTRL, self.step_controller)

        self.counter = 0

    def foot_left_contact_callback(self, msg: StampedBool):
        self.foot_left_contact = msg.data

    def foot_right_contact_callback(self, msg: StampedBool):
        self.foot_right_contact = msg.data

    def urdf_callback(self, msg: String):
        """Extracts all joint names along with their min and max limits from the URDF."""
        urdf_str = msg.data
        try:
            root = ET.fromstring(urdf_str)
            for joint in root.findall('joint'):
                name = joint.get('name')
                limit = joint.find('limit')
                min_limit = limit.get('lower') if limit is not None else "N/A"
                max_limit = limit.get('upper') if limit is not None else "N/A"
                if min_limit == "N/A" or max_limit == "N/A":
                    self.get_logger().warn(f"Joint {name} has no limits. Will not add to the dictionary.")
                else:
                    self.joints_from_urdf[name] = (min_limit, max_limit)
        except ET.ParseError as e:
            self.get_logger().error(f"Failed to parse URDF: {e}")
        self.get_logger().info(f'Extracted Joints: {self.joints_from_urdf}')
        assert len(self.joints_from_urdf.keys()) == self.action_size, \
            f"Number of joints in URDF ({len(self.joints_from_urdf.keys())}) does not match the action size ({self.action_size})."

    def odom_cb(self, msg: Odometry):
        with self.lock:
            self.odom_msg = msg

    def imu_cb(self, msg: Imu):
        with self.lock:
            self.imu_msg = msg

    # def gravity_cb(self, msg: Vector3Stamped):
    #     with self.lock:
    #         self.gravity_msg = msg

    def joint_states_cb(self, msg: JointState):
        with self.lock:
            self.joints_msg = msg

    def run_ppo_ctrl(self):
        ''' Runs the PPO controller. '''
        lin_vel_B = np.array([self.odom_msg.twist.twist.linear.x, self.odom_msg.twist.twist.linear.y, self.odom_msg.twist.twist.linear.z])
        gyro = np.array([self.imu_msg.angular_velocity.x, self.imu_msg.angular_velocity.y, self.imu_msg.angular_velocity.z])

        quat = self.odom_msg.pose.pose.orientation
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        rot_matrix = r.as_matrix()
        gravity_v2 = rot_matrix.T @ np.array([0, 0, -1])
        # gravity = np.array([self.gravity_msg.vector.x, self.gravity_msg.vector.y, self.gravity_msg.vector.z])

        joints_pos = []
        joints_vel = []
        for joint_name in self.joint_names_PPO:
            if joint_name in self.joints_msg.name:
                joints_pos.append(self.joints_msg.position[self.joints_msg.name.index(joint_name)])
                joints_vel.append(self.joints_msg.velocity[self.joints_msg.name.index(joint_name)])
            else:
                joints_pos.append(0)
                joints_vel.append(0)

        phase_tp1 = self.info["phase"] + self.info["phase_dt"]
        self.info["phase"] = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        cos = np.cos(self.info["phase"])
        sin = np.sin(self.info["phase"])
        phase = np.concatenate([cos, sin])

        command = np.array([0.1, 0.0, 0.0])
        input_ppo = np.hstack([
            lin_vel_B,   # 3
            gyro,     # 3
            gravity_v2,  # 3
            command,  # 3
            joints_pos - self.default_q_joints,  # 10
            joints_vel,  # 10
            self.last_action,  # 10
            phase,
        ])

        self.obs = {
            'privileged_state': jp.zeros(self.config['observation_size']['privileged_state']),
            'state': jp.array(input_ppo)
        }

        act_rng, self.rng = jax.random.split(self.rng)
        action_ppo, _ = self.jit_policy(self.obs, act_rng)
        action_ppo_np = np.array(action_ppo)
        motor_targets = self.default_q_joints + action_ppo_np * 0.5
        self.publish_joints(motor_targets)
        self.last_action = action_ppo_np.copy()

    def step_controller(self):
        time_now = self.get_clock().now().nanoseconds / 1e9
        if self.joints_from_urdf == {}:
            self.get_logger().error('Joint limits not set. Cannot publish trajectory.')
            return

        # Check if we have all the data.
        if self.imu_msg is None:
            self.get_logger().warn('IMU data not received. Skipping this step.')
            return

        if self.joints_msg is None:
            self.get_logger().warn('Joint data not received. Skipping this step.')
            return

        if self.odom_msg is None:
            self.get_logger().warn('Odometry data not received. Skipping this step.')
            return

        # if self.gravity_msg is None:
        #     self.get_logger().warn('Gravity data not received. Skipping this step.')
        #     return

        # # Check for old data.
        imu_time = self.imu_msg.header.stamp.sec + self.imu_msg.header.stamp.nanosec / 1e9
        if abs(time_now - imu_time) > 0.1:
            self.get_logger().warn('timenow {}, imu time {}'.format(time_now, self.imu_msg.header.stamp.sec))
            self.get_logger().warn('IMU data is old. Skipping this step.')
            return

        joints_state_time = self.joints_msg.header.stamp.sec + self.joints_msg.header.stamp.nanosec / 1e9
        if abs(time_now - joints_state_time) > 0.1:
            self.get_logger().warn('Joint state data is old. Skipping this step.')
            return

        odom_time = self.odom_msg.header.stamp.sec + self.odom_msg.header.stamp.nanosec / 1e9
        if abs(time_now - odom_time) > 0.1:
            self.get_logger().warn('Odometry data is old. Skipping this step.')
            return

        # gravity_time = self.gravity_msg.header.stamp.sec + self.gravity_msg.header.stamp.nanosec / 1e9
        # if abs(time_now - gravity_time) > 0.1:
        #     self.get_logger().warn('Gravity data is old. Skipping this step.')
        #     return

        if (self.foot_left_contact == False and self.foot_right_contact == False):
            self.timeout_for_no_feet_in_contact -= DT_CTRL
        else:
            self.timeout_for_no_feet_in_contact = TIME_NO_FEET_IN_CONTACT
            self.state = "FOOT_IN_CONTACT"
            self.initialization_done = True

        if (self.timeout_for_no_feet_in_contact < 0):
            self.get_logger().info("No feet in contact for too long")
            if (self.state == "FOOT_IN_CONTACT"):
                self.get_logger().info("Switching to INIT")
                self.state = "INIT"
                self.initialization_done = False
                self.t_init_traj_ = 0

        if (self.state == "INIT"):
            self.t_init_traj = 0.0
            self.state = "RAMP_TO_STARTING_POS"

        if (self.state == "RAMP_TO_STARTING_POS"):
            self.publish_joints(self.start_q_joints)
            self.t_init_traj += DT_CTRL
            if (self.t_init_traj >= 2.0): # give it some time to ramp up
                self.initialization_done = True

        if (self.state == "FOOT_IN_CONTACT" and self.initialization_done == True):
            time_now = self.get_clock().now().nanoseconds / 1e9
            self.run_ppo_ctrl()
            # self.get_logger().info('Running PPO controller')
            dt_ctrl = self.get_clock().now().nanoseconds / 1e9 - time_now
            if abs(dt_ctrl) > DT_CTRL:
                self.get_logger().warn(f'Controller took too long: {dt_ctrl} s')

    def publish_joints(self, joints: list):
        ''' Publishes the joint angles to the robot. '''
        msg = JointTrajectory()
        msg.joint_names = self.joint_names_PPO
        msg.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()

        # Ensure the limits are satisfied.
        joints_out = joints.copy()
        for i, joint_name in enumerate(self.joint_names_PPO):
            min_limit, max_limit = self.joints_from_urdf[joint_name]
            if joints_out[i] < float(min_limit):
                self.get_logger().warn(f'Joint {joint_name} is below the min limit {min_limit}. It wants to be {joints_out[i]}. Setting to min limit')
                joints_out[i] = float(min_limit)
            if joints_out[i] > float(max_limit):
                self.get_logger().warn(f'Joint {joint_name} is above the max limit {max_limit}. It wants to be {joints_out[i]}. Setting to max limit')
                joints_out[i] = float(max_limit)

        point.positions = joints_out
        point.velocities = [0.0] * len(joints_out)
        point.effort = [0.0] * len(joints_out)
        msg.points.append(point)
        self.publisher_joints.publish(msg)
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()