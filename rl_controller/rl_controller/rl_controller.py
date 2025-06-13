from trajectory_msgs.msg import JointTrajectory
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from biped_bringup.msg import StampedBool
from geometry_msgs.msg import TwistStamped
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
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter

# Check JAX devices and set the CPU.
jax.config.update('jax_platform_name', 'cpu')
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
print("Available devices:", jax.devices())
print(str(jax.local_devices()[0]))

# Load results folder path from environment variable with fallback
POLICY_PATH = os.getenv('POLICY_PATH',
                        '~/.ros/policy')

class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher_rl')

        # Parameters.
        self.declare_parameter("low_torque", rclpy.parameter.Parameter.Type.DOUBLE)
        self.declare_parameter("high_torque", rclpy.parameter.Parameter.Type.DOUBLE)
        self.declare_parameter("time_init_traj", rclpy.parameter.Parameter.Type.DOUBLE)
        self.declare_parameter("time_no_feet_in_contact", rclpy.parameter.Parameter.Type.DOUBLE)

        self.use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self.low_torque = self.get_parameter("low_torque").get_parameter_value().double_value
        self.high_torque = self.get_parameter("high_torque").get_parameter_value().double_value
        self.time_init_traj = self.get_parameter("time_init_traj").get_parameter_value().double_value
        self.time_no_feet_in_contact = self.get_parameter("time_no_feet_in_contact").get_parameter_value().double_value

        # Load PPO policy.
        self.get_logger().info('Loading PPO policy...')
        latest_results_folder = sorted(os.listdir(POLICY_PATH))[-1]
        folders = sorted(os.listdir(epath.Path(POLICY_PATH) / latest_results_folder))
        folders = [f for f in folders if os.path.isdir(epath.Path(POLICY_PATH) / latest_results_folder / f)]
        if len(folders) == 0:
            raise ValueError(f'No folders found in {epath.Path(POLICY_PATH) / latest_results_folder}')
        if len(folders) > 1:
            latest_weights_folder = folders[-1]
        else:
            latest_weights_folder = folders
        self.get_logger().info(f'    Latest weights folder: {latest_weights_folder}')
        path = epath.Path(POLICY_PATH) / latest_results_folder / latest_weights_folder
        self.get_logger().info(f'    Loading policy from: {path}')

        # Go through the sharding and replace the CUDA with CPU.
        for shard in os.listdir(path):
            if shard.endswith('sharding'):
                with open(os.path.join(path, shard), 'r') as f:
                    content = f.read()
                    print(content)
                    content = content.replace('cuda:0', 'TFRT_CPU_0')
                    with open(os.path.join(path, shard), 'w') as f:
                        f.write(content)
        self.policy_fn = ppo_checkpoint.load_policy(path)
        self.jit_policy = jax.jit(self.policy_fn)
        self.rng = jax.random.PRNGKey(1)

        # Actuator mapping.
        actuator_mapping_PPO_file = epath.Path(POLICY_PATH) / latest_results_folder / 'policy_actuator_mapping.json'
        with open(actuator_mapping_PPO_file) as f:
            self.actuator_mapping_PPO = json.load(f)
        self.actuator_mapping_PPO = self.actuator_mapping_PPO['actuated_joint_names_to_policy_idx_dict']

        # Load params of the PPO policy.
        network_config_file_path = epath.Path(POLICY_PATH) / latest_results_folder / 'ppo_network_config.json'
        with open(network_config_file_path) as f:
            self.network_config = json.load(f)
        self.get_logger().info(f'Network config: {self.network_config}')
        self.action_size = self.network_config['action_size']
        self.obs = {
            'privileged_state': jp.zeros(self.network_config['observation_size']['privileged_state']),
            'state': jp.zeros(self.network_config['observation_size']['state'])
        }

        # Initialize state history
        self.state_history = None
        self.state_size = self.network_config['observation_size']['state'][0]

        # Config default joint angles.
        default_joint_angles_file = epath.Path(POLICY_PATH) / latest_results_folder / 'initial_qpos.json'
        with open(default_joint_angles_file) as f:
            self.default_q_joints = json.load(f)
            # Remove root joint.
            self.default_q_joints = {k: v for k, v in self.default_q_joints.items() if k != 'root'}
        self.get_logger().info(f'Default joint angles: {self.default_q_joints}')
        # Configs for the controller.
        configs_training = epath.Path(POLICY_PATH) / latest_results_folder / 'config.json'
        with open(configs_training) as f:
            self.configs_training = json.load(f)
        self.get_logger().info(f'Configs training: {self.configs_training}')

        self.dt_ctrl = self.configs_training['ctrl_dt']
        self.history_len = self.configs_training['history_len']

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
        self.state = "RAMP_TO_STARTING_POS"

        # Lock.
        self.lock = threading.Lock()

        # Params update.
        if self.use_sim_time == False:
            self.moteus_set_param = self.create_client(SetParameters, '/moteus/set_parameters')
            while not self.moteus_set_param.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.moteus_param_requests = []
            self.update_moteus_parameter('global_max_torque', self.high_torque)

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

        self.vel_cmd_x = 0.0
        self.vel_cmd_y = 0.0
        self.vel_cmd_sub = None
        self.vel_cmd_sub = self.create_subscription(
            TwistStamped,
            '~/vel_cmd',
            self.vel_cmd_cb,
            1)

        gait_freq = 1.5
        phase_dt = 2 * np.pi * self.dt_ctrl * gait_freq
        phase = np.array([0, np.pi])

        self.info = {
            'phase': phase,
            'phase_dt': phase_dt,
        }

        self.last_action = np.zeros(self.action_size)

        self.start_q_joints = self.default_q_joints.copy()
        self.timeout_for_no_feet_in_contact = 0.0

        # TODO: put _compensated in the name of the topic for hardware.
        self.publisher_joints = self.create_publisher(JointTrajectory, '~/joint_trajectory', 10)
        self.publisher_joints_ppo = self.create_publisher(JointTrajectory, '~/joint_trajectory_ppo', 10)
        self.timer = self.create_timer(self.dt_ctrl, self.step_controller)


    def update_moteus_parameter(self, name_param, value):
        req = SetParameters.Request()
        req.parameters = [Parameter(name=name_param, value=value).to_parameter_msg()]
        self.moteus_param_requests.append(self.moteus_set_param.call_async(req))

    def check_moteus_param_requests(self):
        requests_open = []
        for i, req in enumerate(self.moteus_param_requests):
            if req.done():
                try:
                    response = req.result()
                except Exception as e:
                    self.get_logger().error(f'Service call failed {str(e)}')
                else:
                    self.get_logger().info(f'Parameter updated')
            else:
                requests_open.append(req)
        self.moteus_param_requests = requests_open

    def foot_left_contact_callback(self, msg: StampedBool):
        self.foot_left_contact = msg.data

    def foot_right_contact_callback(self, msg: StampedBool):
        self.foot_right_contact = msg.data

    def vel_cmd_cb(self, msg: TwistStamped):
        self.vel_cmd_x = msg.twist.linear.x
        self.vel_cmd_y = msg.twist.linear.y

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
        # assert len(self.joints_from_urdf.keys()) == self.action_size, \
        #     f"Number of joints in URDF ({len(self.joints_from_urdf.keys())}) does not match the action size ({self.action_size})."

    def odom_cb(self, msg: Odometry):
        with self.lock:
            self.odom_msg = msg

    def imu_cb(self, msg: Imu):
        with self.lock:
            self.imu_msg = msg

    def joint_states_cb(self, msg: JointState):
        with self.lock:
            self.joints_msg = msg

    def run_ppo_ctrl(self):
        ''' Runs the PPO controller. '''
        time_now = self.get_clock().now().nanoseconds / 1e9
        # Add robustness to run only when odom is available.
        if self.odom_msg is None:
            self.get_logger().warn('Odometry data not received. Skipping this step.')
            return

        # Add robustness to run only when imu is available.
        if self.imu_msg is None:
            self.get_logger().warn('IMU data not received. Skipping this step.')
            return

        # Add robustness to run only when joints are available.
        if self.joints_msg is None:
            self.get_logger().warn('Joint data not received. Skipping this step.')
            return

        # Old messages.
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

        # Linear velocity.
        lin_vel_B = np.array([self.odom_msg.twist.twist.linear.x,
                             self.odom_msg.twist.twist.linear.y,
                             self.odom_msg.twist.twist.linear.z])
        # Angular velocity.
        gyro = np.array([self.imu_msg.angular_velocity.x,
                        self.imu_msg.angular_velocity.y,
                        self.imu_msg.angular_velocity.z])

        # Up vector in body frame.
        quat = self.odom_msg.pose.pose.orientation
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        rot_matrix = r.as_matrix()
        up_B = rot_matrix.T @ np.array([0, 0, 1])

        # Joints position and velocity.
        joints_pos = []
        joints_vel = []
        for joint_name in self.default_q_joints.keys():
            # Get the joint position and velocity from the joint state message.
            if joint_name in self.joints_msg.name:
                joints_pos.append(self.joints_msg.position[self.joints_msg.name.index(joint_name)])
                joints_vel.append(self.joints_msg.velocity[self.joints_msg.name.index(joint_name)])
            else:
                joints_pos.append(0)
                joints_vel.append(0)

        # Phase.
        phase_tp1 = self.info["phase"] + self.info["phase_dt"]
        self.info["phase"] = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        cos = np.cos(self.info["phase"])
        sin = np.sin(self.info["phase"])
        phase = np.concatenate([cos, sin])

        # Command.
        command = np.array([self.vel_cmd_x, self.vel_cmd_y, 0.0])

        # Input to the PPO policy.
        current_state = np.hstack([
            lin_vel_B,   # 3
            gyro,     # 3
            up_B,  # 3
            command,  # 3
            joints_pos - np.array(list(self.default_q_joints.values())),  # 10
            joints_vel,  # 10
            self.last_action,  # 8
            phase,
        ])

        # Initialize state history if needed.
        if self.state_history is None:
            self.get_logger().info(f'Initializing state history with shape: {(self.history_len, current_state.shape[0])}')
            self.state_history = np.zeros((self.history_len, current_state.shape[0]))

        # Update state history.
        self.state_history = np.roll(self.state_history, -1, axis=0)
        self.state_history[-1] = current_state

        self.obs = {
            'privileged_state': jp.zeros(self.network_config['observation_size']['privileged_state']),
            'state': jp.array(self.state_history.ravel())
        }

        act_rng, self.rng = jax.random.split(self.rng)
        action_ppo, _ = self.jit_policy(self.obs, act_rng)
        action_ppo_np = np.array(action_ppo)

        # Map the action to the joint names.
        motor_targets = self.default_q_joints.copy()
        motor_targets_ppo = {}
        for joint_name, idx in self.actuator_mapping_PPO.items():
            if idx is not None:  # Skip None values (like for ANKLE joints)
                if joint_name == 'L_YAW' or joint_name == 'R_YAW':
                    action_ppo_np[idx] = 0.0
                motor_targets[joint_name] += action_ppo_np[idx]
                motor_targets_ppo[joint_name] = action_ppo_np[idx]
        self.publish_joints(motor_targets)
        self.publish_ppo_residual_joints(motor_targets_ppo)

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

        feet_in_contact = self.foot_left_contact or self.foot_right_contact
        if not feet_in_contact:
            self.timeout_for_no_feet_in_contact -= self.dt_ctrl
        else:
            self.timeout_for_no_feet_in_contact = self.time_no_feet_in_contact

        if self.state == "RAMP_TO_STARTING_POS":
            if feet_in_contact:
                self.state = "WALKING"
                if self.use_sim_time == False:
                    self.get_logger().info("Feet in contact. Setting high torque.")
                    self.update_moteus_parameter('global_max_torque', self.high_torque)

        elif self.state == "WALKING":
            if self.timeout_for_no_feet_in_contact < 0:
                self.get_logger().info("No feet in contact for too long")
                self.state = "RAMP_TO_STARTING_POS"
                if self.use_sim_time == False:
                    self.get_logger().info("No feet in contact for too long. Setting low torque.")
                    self.update_moteus_parameter('global_max_torque', self.low_torque)

        if self.state == "RAMP_TO_STARTING_POS":
            self.publish_joints(self.start_q_joints)
        elif self.state == "WALKING":
            time_now = self.get_clock().now().nanoseconds / 1e9
            self.run_ppo_ctrl()
            dt_ctrl = self.get_clock().now().nanoseconds / 1e9 - time_now
            if abs(dt_ctrl) > self.dt_ctrl:
                self.get_logger().warn(f'Controller took too long: {dt_ctrl} s')

    def publish_joints(self, joints: dict):
        ''' Publishes the joint angles to the robot. '''
        msg = JointTrajectory()
        msg.joint_names = list(joints.keys())
        msg.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()

        # Ensure the limits are satisfied.
        joints_out = []
        for joint_name in joints.keys():
            value = joints[joint_name]
            min_limit, max_limit = self.joints_from_urdf[joint_name]
            if value < float(min_limit):
                self.get_logger().warn(f'Joint {joint_name} is below the min limit {min_limit}. It wants to be {value}. Setting to min limit')
                value = float(min_limit)
            if value > float(max_limit):
                self.get_logger().warn(f'Joint {joint_name} is above the max limit {max_limit}. It wants to be {value}. Setting to max limit')
                value = float(max_limit)
            joints_out.append(value)

        point.positions = joints_out
        point.velocities = [0.0] * len(joints_out)
        point.effort = [0.0] * len(joints_out)
        msg.points.append(point)
        self.publisher_joints.publish(msg)

    def publish_ppo_residual_joints(self, joints_ppo: dict):
        msg = JointTrajectory()
        msg.joint_names = list(joints_ppo.keys())
        msg.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()
        point.positions = list(joints_ppo.values())
        msg.points.append(point)
        self.publisher_joints_ppo.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()