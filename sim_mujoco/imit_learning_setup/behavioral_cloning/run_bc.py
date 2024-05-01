import os
import numpy as np
import pandas as pd

from bc import bc

data_rel_paths = [
    # "../../sim_mujoco/data/dataset_backwards.csv", "../../sim_mujoco/data/dataset_forward_sideways.csv", "../../sim_mujoco/data/dataset_misc.csv"
    # "../../sim_mujoco/data/in_place.csv", "../../sim_mujoco/data/in_place_long.csv"
    # "../../sim_mujoco/data/in_place.csv"
    "../data_processing/updated_data/in_place_long_v4.csv"
]
# TODO: train
data_paths = [os.path.join(os.path.dirname(os.path.realpath(__file__)), pth) for pth in data_rel_paths]
policy_name = "bc_v6_normalized_data_long_cleaned_ignore_foot"
# Fraction of data to train on. If you are going to test the policy on the biped in sim, use 1. (no reason to leave any data out)
train_frac = 0.9


# state_columns = [
#     "L_YAW_pos", "L_HAA_pos", "L_HFE_pos", "L_KFE_pos", "L_ANKLE_pos",
#     "R_YAW_pos", "R_HAA_pos", "R_HFE_pos", "R_KFE_pos", "R_ANKLE_pos",
#     "L_YAW_vel", "L_HAA_vel", "L_HFE_vel", "L_KFE_vel", "L_ANKLE_vel",
#     "R_YAW_vel", "R_HAA_vel", "R_HFE_vel", "R_KFE_vel", "R_ANKLE_vel", 
#     "vel_x_BF", "vel_y_BF", "vel_z_BF", "normal_vec_x_BF", "normal_vec_y_BF", "normal_vec_z_BF", 
#     "omega_x", "omega_y", "omega_z", "vx_des_BF", "vy_des_BF", 
#     "right_foot_t_since_contact", "right_foot_t_since_no_contact", 
#     "right_foot_pos_x_BF", "right_foot_pos_y_BF", "right_foot_pos_z_BF",
#     "left_foot_t_since_contact", "left_foot_t_since_no_contact",
#     "left_foot_pos_x_BF", "left_foot_pos_y_BF", "left_foot_pos_z_BF"
# ]

# "right_foot_pos_z_BF" & "right_foot_pos_x_BF" & "right_foot_pos_y_BF" is removed right now since it is always zero in current data.
state_columns = [
    "L_YAW_pos", "L_HAA_pos", "L_HFE_pos", "L_KFE_pos", "L_ANKLE_pos",
    "R_YAW_pos", "R_HAA_pos", "R_HFE_pos", "R_KFE_pos", "R_ANKLE_pos",
    "L_YAW_vel", "L_HAA_vel", "L_HFE_vel", "L_KFE_vel", "L_ANKLE_vel",
    "R_YAW_vel", "R_HAA_vel", "R_HFE_vel", "R_KFE_vel", "R_ANKLE_vel", 
    "vel_x_BF", "vel_y_BF", "vel_z_BF", "normal_vec_x_BF", "normal_vec_y_BF", "normal_vec_z_BF", 
    "omega_x", "omega_y", "omega_z", "vx_des_BF", "vy_des_BF", 
    # "right_foot_t_since_contact", "right_foot_t_since_no_contact", 
    # "right_foot_t_since_contact",
    # "left_foot_t_since_contact", "left_foot_t_since_no_contact",
    # "left_foot_t_since_contact",
    "left_foot_pos_x_BF", "left_foot_pos_y_BF", "left_foot_pos_z_BF"
]
action_columns = [
    # "L_YAW_tau_ff", "L_HAA_tau_ff", "L_HFE_tau_ff", "L_KFE_tau_ff", "L_ANKLE_tau_ff",
    "L_YAW_tau_ff", "L_HFE_tau_ff", "L_KFE_tau_ff", "L_ANKLE_tau_ff",
    # "R_YAW_tau_ff", "R_HAA_tau_ff", "R_HFE_tau_ff", "R_KFE_tau_ff", "R_ANKLE_tau_ff",
    "R_YAW_tau_ff", "R_HFE_tau_ff", "R_KFE_tau_ff", "R_ANKLE_tau_ff",
    "L_YAW_q_des", "L_HAA_q_des", "L_HFE_q_des", "L_KFE_q_des", "L_ANKLE_q_des",
    "R_YAW_q_des", "R_HAA_q_des", "R_HFE_q_des", "R_KFE_q_des", "R_ANKLE_q_des",
    "L_YAW_q_vel_des", "L_HAA_q_vel_des", "L_HFE_q_vel_des", "L_KFE_q_vel_des", "L_ANKLE_q_vel_des",
    "R_YAW_q_vel_des", "R_HAA_q_vel_des", "R_HFE_q_vel_des", "R_KFE_q_vel_des", "R_ANKLE_q_vel_des"
]

# how about changing the layer node to 32?
use_spectral_norm = True
num_input_states = 3 # Number of states to include in input
layer1 = 256
layer2 = 512 
layer3 = 512
layer4 = 256
policy_arch = [
    {'Layer': 'Linear', 'Input': len(state_columns) * num_input_states, 'Output': layer1, 'SpectralNorm': use_spectral_norm},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': layer1, 'Output': layer2, 'SpectralNorm': use_spectral_norm},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': layer2, 'Output': layer3, 'SpectralNorm': use_spectral_norm},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': layer3, 'Output': layer4, 'SpectralNorm': use_spectral_norm},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': layer4, 'Output': len(action_columns), 'SpectralNorm': use_spectral_norm}
]
train_epochs = 10

def main():
    # Load Data
    dataset = pd.DataFrame()
    for dp in data_paths:
        ds = pd.read_csv(dp)
        dataset = pd.concat((dataset, ds))
    num_steps = dataset.shape[0]

    # randomly shuffle all the rows.
    # dataset = dataset.sample(frac=1).reset_index(drop=True)

    states = dataset[state_columns].to_numpy(dtype=np.float64)
    actions = dataset[action_columns].to_numpy(dtype=np.float64)
    # For debugging & finding std of each state and removing states with std of 0
    # print(np.std(states,0))
    # print(np.where(np.std(states,0) == 0.0))
    # print(state_columns[32])
    
    # Stack states if more than one included in input
    states = np.hstack([states[ii: states.shape[0] - (num_input_states - ii - 1), :] for ii in range(num_input_states)])
    actions = actions[num_input_states - 1:, :]
    
    # Segment out some data for testing
    train_states = states[:int(num_steps * train_frac), :]
    train_actions = actions[:int(num_steps * train_frac), :]

    # Run vanilla behavior cloning
    behavior_clone = bc(train_states, train_actions, policy_arch)
    behavior_clone.train(train_epochs, use_best_weights=False)

    behavior_clone.save_policy(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"policies/{policy_name}.pt"))
    

if __name__ == "__main__":
    main()