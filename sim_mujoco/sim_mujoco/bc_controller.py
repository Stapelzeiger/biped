import numpy as np
import torch
from pickle import load

import sys
sys.path.insert(0, '/home/zhonghezheng13579/ros_ws/src/biped/sim_mujoco/imit_learning_setup/behavioral_cloning/')
from bc import BC_Agent

class PolicyBC:
    def __init__(self, state_size, action_size, num_input_states):
        self.state_size = state_size
        self.action_size = action_size
        self.folder = '/home/leo/biped_ws/src/biped/sim_mujoco/imit_learning_setup/behavioral_cloning/' # todo fix this so it's not hardcoded
        # self.policy_name = 'bc_policy_4_500e_no_spectral_norm'
        self.policy_name = 'v1_6_12_24_no_control'

        num_input_states = num_input_states # Number of states to include in input
        use_spectral_norm = True
        layer1 = 256 * 1
        layer2 = 512 * 1
        layer3 = 512 * 1
        layer4 = 256 * 1
        policy_arch = [
            {'Layer': 'Linear', 'Input': self.state_size * num_input_states, 'Output': layer1, 'SpectralNorm': use_spectral_norm},
            {'Layer': 'ReLU'},
            {'Layer': 'Linear', 'Input': layer1, 'Output': layer2, 'SpectralNorm': use_spectral_norm},
            {'Layer': 'ReLU'},
            {'Layer': 'Linear', 'Input': layer2, 'Output': layer3, 'SpectralNorm': use_spectral_norm},
            {'Layer': 'ReLU'},
            {'Layer': 'Linear', 'Input': layer3, 'Output': layer4, 'SpectralNorm': use_spectral_norm},
            {'Layer': 'ReLU'},
            {'Layer': 'Linear', 'Input': layer4, 'Output': self.action_size, 'SpectralNorm': use_spectral_norm}
        ]

        self.policy = BC_Agent(policy_arch, torch.zeros(num_input_states*self.state_size),
                                            torch.zeros(num_input_states*self.state_size),
                                            torch.zeros(self.action_size),
                                            torch.zeros(self.action_size))
        self.policy.load_state_dict(torch.load(self.folder + f"policies/{self.policy_name}.pt"))

        with open("/home/leo/biped_ws/src/biped/sim_mujoco/imit_learning_setup/data_processing/updated_data/action_scaler_just_walking_v1.pkl", "rb") as f:
            self.scaler_a = load(f)

        with open("/home/leo/biped_ws/src/biped/sim_mujoco/imit_learning_setup/data_processing/updated_data/state_scaler_just_walking_v1.pkl", "rb") as f:
            self.scaler_s = load(f)


    def __call__(self, input_state):
        input_state = np.reshape(input_state, (1, -1))
        # since the input will be 3x, the quantile transformer just takes current state, so we need to reshape it and pass it in.
        # not sure if this it best method.
        # convert 123 features to 41x3 different calls to transform
        arr1 = input_state[:, :41]
        arr2 = input_state[:, 41:82]
        arr3 = input_state[:, 82:123]

        scaled_arr1 = self.scaler_s.transform(arr1)
        scaled_arr2 = self.scaler_s.transform(arr2)
        scaled_arr3 = self.scaler_s.transform(arr3)
        scaled_input_state = np.concatenate((scaled_arr1, scaled_arr2, scaled_arr3), axis=1)

        # reshape into an 2d array since that is how we trained it.
        policy_actions = self.policy.get_action(torch.tensor(scaled_input_state, dtype=torch.float32))
        
        unscaled_policy_actions = self.scaler_a.inverse_transform(policy_actions.detach().to('cpu').numpy())
        # if on gpu, then convert back to cpu, if on cpu, then eh
        unscaled_policy_actions = np.reshape(unscaled_policy_actions, (-1))
        return unscaled_policy_actions
