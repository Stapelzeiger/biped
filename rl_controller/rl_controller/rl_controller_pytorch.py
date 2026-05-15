import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path
from etils import epath
sys.path.append(
    str(Path(__file__).resolve().parents[2] / "robot_learning" / "robot_learning" / "src")
)
from jax_simplified.cleanrl_ppo import Agent


class RL_Controller:
    def __init__(self, results_folder: str,
                priviliged_state_size: int,
                state_size: int,
                action_size: int):

        # Checkpoint path.
        self.checkpoint_path = results_folder / 'checkpoints' / 'best_agent.pt'

        # Initialize the agent.
        self.agent = Agent(priviliged_state_size,
                        state_size,
                        action_size)

        self.load_policy()

    def load_policy(self):
        self.agent.load_checkpoint(self.checkpoint_path)

    def run(self, obs):
        """``obs`` is either policy state array or ``{'state': ..., 'privileged_state': ...}`` (actor uses ``state`` only)."""
        x = obs["state"] if isinstance(obs, dict) else obs
        # Reverses / fancy indexing can yield negative strides; PyTorch rejects those views.
        x = torch.as_tensor(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
        with torch.no_grad():
            action = self.agent.get_action_deterministic(x)
        return action.squeeze(0).cpu().numpy()
