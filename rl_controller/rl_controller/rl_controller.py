import os
import shutil
import jax
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import numpy as np
import sys
import os
import json
from etils import epath
import jax.numpy as jp
from scipy.spatial.transform import Rotation as R

class RL_Controller:
    def __init__(self, path):
        self.path = path

        # Go through the sharding and replace the CUDA with CPU.
        for shard in os.listdir(path):
            if shard.endswith('sharding'):
                with open(os.path.join(path, shard), 'r') as f:
                    content = f.read()
                    print(content)
                    content = content.replace('cuda:0', 'TFRT_CPU_0')
                    with open(os.path.join(path, shard), 'w') as f:
                        f.write(content)
        # Copy the file ppo_network_config.json one back from the path
        shutil.copy(os.path.join(path, 'ppo_network_config.json'), os.path.join(path, '..', 'ppo_network_config.json')) # Required for the ppo_checkpoint.load_policy
        self.rng = jax.random.PRNGKey(0)
        self.policy_fn = ppo_checkpoint.load_policy(path)
        self.jit_policy = jax.jit(self.policy_fn)

    def run(self, state):
        act_rng, self.rng = jax.random.split(self.rng)
        action_ppo, _ = self.jit_policy(state, act_rng)
        action_ppo_np = np.array(action_ppo)

        return action_ppo_np


def main():
    sys.path.append("../../sim_mujoco/sim_mujoco/submodules")

    # Load the biped.
    from mujoco_env_sim import Biped
    biped = Biped(xml="../../biped_robot_description/urdf/biped_RL_v2.xml",
                  use_RL=True,
                  visualize_mujoco=True)

    # Initialize the biped.
    q_init = np.array(biped.model.keyframe("home").qpos)
    biped.data.qpos = q_init
    biped.data.qvel = np.zeros(biped.model.nv)
    import mujoco as mj
    mj.mj_step(biped.model, biped.data)
    # biped.init(q=q_init)

    # Load PPO policy.
    POLICY_PATH = os.path.abspath("../policy_path/")
    print('Loading PPO policy...')
    latest_results_folder = sorted(os.listdir(POLICY_PATH))[-1]
    folders = sorted(os.listdir(epath.Path(POLICY_PATH) / latest_results_folder))
    folders = [f for f in folders if os.path.isdir(epath.Path(POLICY_PATH) / latest_results_folder / f)] # Only keep the folders that are directories.
    if len(folders) == 0:
        raise ValueError(f'No folders found in {epath.Path(POLICY_PATH) / latest_results_folder}')
    if len(folders) > 1:
        latest_weights_folder = folders[-1]
    else:
        latest_weights_folder = folders
    print(f'    Latest weights folder: {latest_weights_folder}')
    path = epath.Path(POLICY_PATH) / latest_results_folder / latest_weights_folder
    print(f'    Loading policy from: {path}')

    # Initialize the RL controller.
    rl_controller = RL_Controller(path=path)

    # Actuator mapping.
    actuator_mapping_PPO_file = epath.Path(POLICY_PATH) / latest_results_folder / 'policy_actuator_mapping.json'
    with open(actuator_mapping_PPO_file) as f:
        actuator_mapping_PPO = json.load(f)
    actuator_mapping_PPO = actuator_mapping_PPO['actuated_joint_names_to_policy_idx_dict']

    # Load params of the PPO policy.
    network_config_file_path = epath.Path(POLICY_PATH) / latest_results_folder / latest_weights_folder / 'ppo_network_config.json'
    with open(network_config_file_path) as f:
        network_config = json.load(f)
    print(f'Network config: {network_config}')
    action_size = network_config['action_size']

    # Initialize state history.
    state_history = None

    # Config default joint angles.
    default_joint_angles_file = epath.Path(POLICY_PATH) / latest_results_folder / 'initial_qpos.json'
    with open(default_joint_angles_file) as f:
        default_q_joints = json.load(f)
        # Remove root joint.
        default_q_joints = {k: v for k, v in default_q_joints.items() if k != 'root'}
    print(f'Default joint angles: {default_q_joints}')
    # Configs for the controller.
    configs_training = epath.Path(POLICY_PATH) / latest_results_folder / 'config.json'
    with open(configs_training) as f:
        configs_training = json.load(f)
    print(f'Configs training: {configs_training}')

    dt_ctrl = configs_training['ctrl_dt']
    dt_sim = configs_training['sim_dt']
    history_len = configs_training['history_len']

    # Lower the robot until contact is detected.
    step_count = 0
    max_steps = 10000  # Safety limit.
    
    state_history = None
    
    gait_freq = 1.5
    phase_dt = 2 * np.pi * dt_ctrl * gait_freq
    phase = np.array([0, np.pi])

    info = {
        'phase': phase,
        'phase_dt': phase_dt,
    }

    last_action = np.zeros(action_size)
    
    motor_targets = {}
    for joint_name, idx in actuator_mapping_PPO.items():
        if idx is not None:  # Skip None values (like for ANKLE joints)
            motor_targets[joint_name] = {
                'pos': default_q_joints[joint_name],
                'vel': 0.0,
                'effort': 0.0
            }

    counter_for_ctrl = 0

    DT_SIM = configs_training['sim_dt']
    DT_CTRL = configs_training['ctrl_dt']
    N_CTRL = int(DT_CTRL / DT_SIM)

    while step_count < max_steps:
        
        # Set the joint states.
        biped.set_joint_state()

        # Run the PPO controller.
        if counter_for_ctrl % N_CTRL == 0:

            # Get the state.
            q_w = biped.data.qpos[3]
            q_x = biped.data.qpos[4]
            q_y = biped.data.qpos[5]
            q_z = biped.data.qpos[6]
            R_b_to_I = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
            v_b = R_b_to_I.T @ biped.data.qvel[0:3] # linear vel is in inertial frame
            lin_vel_B = np.array([v_b[0], v_b[1], v_b[2]])

            gyro = biped.get_sensor_data(name='gyro')

            # Up vector in body frame.
            r = R.from_quat([q_x, q_y, q_z, q_w])
            rot_matrix = r.as_matrix()
            up_B = rot_matrix.T @ np.array([0, 0, 1])

            # Joints position and velocity.
            joints_pos = []
            joints_vel = []
            for joint_name in default_q_joints.keys():
                # Get the joint position and velocity from the biped
                if joint_name in biped.get_joint_names():
                    joints_pos.append(biped.q_joints[joint_name]['actual_pos'])
                    joints_vel.append(biped.q_joints[joint_name]['actual_vel'])
                    # print(f'Joint name: {joint_name} -> {biped.q_joints[joint_name]["actual_pos"]}')
                else:
                    joints_pos.append(0)
                    joints_vel.append(0)
                
            # Phase.
            phase_tp1 = info["phase"] + info["phase_dt"]
            info["phase"] = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
            cos = np.cos(info["phase"])
            sin = np.sin(info["phase"])
            phase = np.concatenate([cos, sin])

            # Command.
            command = np.array([0.0, 0.0, 0.0])

            # Input to the PPO policy.
            current_state = np.hstack([
                lin_vel_B,   # 3
                gyro,     # 3
                up_B,  # 3
                command,  # 3
                joints_pos - np.array(list(default_q_joints.values())),  # 8
                joints_vel,  # 8
                last_action,  # 8
                phase,  # 2
            ])

            # Initialize state history if needed.
            if state_history is None:
                print(f'Initializing state history with shape: {(history_len, current_state.shape[0])}')
                state_history = np.zeros((history_len, current_state.shape[0]))

            # Update state history.
            state_history = np.roll(state_history, -1, axis=0)
            state_history[-1] = current_state

            obs = {
                'privileged_state': jp.zeros(network_config['observation_size']['privileged_state']),
                'state': jp.array(state_history.ravel())
            }

            action_ppo_np = rl_controller.run(obs)

            # Map the action to the joint names.
            motor_targets = {}
            motor_targets_ppo = {}
            for joint_name, idx in actuator_mapping_PPO.items():
                if idx is not None:  # Skip None values (like for ANKLE joints)
                    motor_targets[joint_name] = {
                        'pos': default_q_joints[joint_name] + action_ppo_np[idx],
                        'vel': 0.0,
                        'effort': 0.0
                    }
                    motor_targets_ppo[joint_name] = action_ppo_np[idx]

            # Update the last action.
            last_action = action_ppo_np.copy()
            counter_for_ctrl = 0
        
        counter_for_ctrl += 1

        biped.step(joint_traj_dict=motor_targets)
        step_count += 1

if __name__ == "__main__":
    main()