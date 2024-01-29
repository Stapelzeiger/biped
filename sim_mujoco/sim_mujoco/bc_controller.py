import numpy as np
import torch

import sys
sys.path.append("/home/leo/biped_ws/src/biped/sim_mujoco/imit_learning_setup/behavioral_cloning/")
from bc import BC_Agent

class PolicyBC:
    def __init__(self, state_size, action_size, num_input_states):
        self.state_size = state_size
        self.action_size = action_size
        self.folder = '/home/leo/biped_ws/src/biped/sim_mujoco/imit_learning_setup/behavioral_cloning/' # todo fix this so it's not hardcoded
        # self.policy_name = 'bc_policy_4_500e_no_spectral_norm'
        self.policy_name = 'bc_policy_4_500e'

        num_input_states = num_input_states # Number of states to include in input
        use_spectral_norm = True
        policy_arch = [
            {'Layer': 'Linear', 'Input': self.state_size * num_input_states, 'Output': 256, 'SpectralNorm': use_spectral_norm},
            {'Layer': 'ReLU'},
            {'Layer': 'Linear', 'Input': 256, 'Output': 512, 'SpectralNorm': use_spectral_norm},
            {'Layer': 'ReLU'},
            {'Layer': 'Linear', 'Input': 512, 'Output': 256, 'SpectralNorm': use_spectral_norm},
            {'Layer': 'ReLU'},
            {'Layer': 'Linear', 'Input': 256, 'Output': self.action_size, 'SpectralNorm': use_spectral_norm}
        ]


        self.policy = BC_Agent(policy_arch, torch.zeros(num_input_states*self.state_size),
                                            torch.zeros(num_input_states*self.state_size),
                                            torch.zeros(self.action_size),
                                            torch.zeros(self.action_size))
        self.policy.load_state_dict(torch.load(self.folder + f"policies/{self.policy_name}.pt"))

    def __call__(self, input_state):
        policy_actions = self.policy.get_action(torch.tensor(input_state, dtype=torch.float32))
        return policy_actions.detach().numpy()
