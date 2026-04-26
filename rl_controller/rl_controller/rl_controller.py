import os
import shutil
import jax
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import numpy as np
import sys
import os
import json
import inspect
from etils import epath
import jax.numpy as jp
from scipy.spatial.transform import Rotation as R
import threading
import time
import mujoco

class RL_Controller:
    @staticmethod
    def _sanitize_network_config(config_path: str):
        """Removes network kwargs unsupported by the installed Brax version."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        network_kwargs = config.get('network_factory_kwargs')
        if not isinstance(network_kwargs, dict):
            return

        valid_keys = set(inspect.signature(ppo_networks.make_ppo_networks).parameters.keys())
        print(f'Valid keys: {valid_keys}')
        filtered_kwargs = {k: v for k, v in network_kwargs.items() if k in valid_keys}
        print(f'Filtered kwargs: {filtered_kwargs}')
        removed_keys = sorted(k for k in network_kwargs.keys() if k not in valid_keys)
        print(f'Removed keys: {removed_keys}')
        if removed_keys:
            print(f"Removing unsupported PPO network kwargs for current Brax: {removed_keys}")
            config['network_factory_kwargs'] = filtered_kwargs
            with open(config_path, 'w') as f:
                json.dump(config, f)

    def __init__(self, path: str):
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
        # Normalize PPO network config for the currently installed Brax API.
        config_file = os.path.join(path, 'ppo_network_config.json')
        self._sanitize_network_config(config_file)
        # Copy the file ppo_network_config.json one back from the path
        copied_config_file = os.path.join(path, '..', 'ppo_network_config.json')
        shutil.copy(config_file, copied_config_file) # Required for the ppo_checkpoint.load_policy
        self._sanitize_network_config(copied_config_file)
        self.rng = jax.random.PRNGKey(1)
        self.policy_fn = ppo_checkpoint.load_policy(path)
        self.jit_policy = jax.jit(self.policy_fn)

    def run(self, state):
        act_rng, self.rng = jax.random.split(self.rng)
        action_ppo, _ = self.jit_policy(state, act_rng)
        action_ppo_np = np.array(action_ppo)

        return action_ppo_np


class KeyboardController:
    def __init__(self):
        self.command = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        self.max_velocity = 1.0  # m/s
        self.max_yaw_rate = 1.0  # rad/s
        self.velocity_step = 0.1  # m/s per key press
        self.yaw_step = 0.1  # rad/s per key press
        self.running = True

        # Key states
        self.keys_pressed = set()

        print("Keyboard controls:")
        print("  Arrow Up/Down: Forward/Backward velocity")
        print("  Arrow Left/Right: Left/Right velocity")
        print("  A/D: Yaw rotation")
        print("  Space: Stop all movement")
        print("  Q: Quit")

    def keyboard_callback(self, keycode):
        """MuJoCo keyboard callback"""
        if keycode == mujoco.viewer.KEY_UP:
            print("Move forward")
            print('hereee')
            self.command[0] = min(self.command[0] + self.velocity_step, self.max_velocity)
        elif keycode == mujoco.viewer.KEY_DOWN:
            print("Move backward")
            self.command[0] = max(self.command[0] - self.velocity_step, -self.max_velocity)
        elif keycode == mujoco.viewer.KEY_LEFT:
            print("Move left")
            self.command[1] = min(self.command[1] + self.velocity_step, self.max_velocity)
        elif keycode == mujoco.viewer.KEY_RIGHT:
            print("Move right")
            self.command[1] = max(self.command[1] - self.velocity_step, -self.max_velocity)
        elif keycode == ord('a') or keycode == ord('A'):
            print("Yaw left")
            self.command[2] = min(self.command[2] + self.yaw_step, self.max_yaw_rate)
        elif keycode == ord('d') or keycode == ord('D'):
            print("Yaw right")
            self.command[2] = max(self.command[2] - self.yaw_step, -self.max_yaw_rate)
        elif keycode == ord(' '):
            print("Stop movement")
            self.command = np.array([0.0, 0.0, 0.0])
        elif keycode == ord('q') or keycode == ord('Q'):
            print("Quitting...")
            self.running = False

    def update_command(self):
        """Update command with decay when no keys are pressed"""
        # Decay velocities when no keys are pressed
        if abs(self.command[0]) > 0.01:
            self.command[0] *= 0.95
        else:
            self.command[0] = 0.0

        if abs(self.command[1]) > 0.01:
            self.command[1] *= 0.95
        else:
            self.command[1] = 0.0

        if abs(self.command[2]) > 0.01:
            self.command[2] *= 0.95
        else:
            self.command[2] = 0.0

    def get_command(self):
        return self.command.copy()

    def stop(self):
        self.running = False
