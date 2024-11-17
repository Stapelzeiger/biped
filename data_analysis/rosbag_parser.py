#!/usr/bin/env python3

# Usage: python3 rosbag_parser.py --path /path/to/rosbag.bag --extract_csvs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import sys
import argparse
import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# Parse the arguments.
parser = argparse.ArgumentParser(description="processes ROS 2 Bag directory path.")
parser.add_argument('--path', type=str, help="ROS 2 Bag directory path")
parser.add_argument('--extract_csvs', action='store_true', help="Extracts the messages to CSV files")
args = parser.parse_args()
rosbag_path = args.path

# List joints.
list_joints = 'L_YAW', 'L_HAA', 'L_HFE', 'L_KFE', 'R_YAW', 'R_HAA', 'R_HFE', 'R_KFE'

# First, extract the messages to CSV files. We save them to the same directory as the rosbag.
if args.extract_csvs:
    rclpy.init()

    reader = SequentialReader()
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    options = StorageOptions(uri=rosbag_path, storage_id='mcap')
    reader.open(options, converter_options)

    # Get the total number of messages for the progress bar
    total_messages = 0
    while reader.has_next():
        reader.read_next()
        total_messages += 1

    reader.open(options, converter_options)

    # Create a dataframe for each joint state.
    header_df_state = 'timestamp', 'name', 'pos', 'vel', 'eff'
    df_state = pd.DataFrame(columns=header_df_state)
    dfs_state = [df_state for _ in range(len(list_joints))]

    # Create a dataframe for each joint trajectory.
    header_df_traj = 'timestamp', 'name', 'pos_des', 'vel_des', 'eff_des'
    df_traj = pd.DataFrame(columns=header_df_traj)
    dfs_traj = [df_traj for _ in range(len(list_joints))]

    t_init = None
    counter = 0.0
    with tqdm(total=total_messages, desc="Processing messages") as pbar:
        while reader.has_next():
            topic, msg, t = reader.read_next()
            if topic == '/joint_trajectory':
                msg = deserialize_message(msg, JointTrajectory)
                for joint_name in list_joints:
                    # Get the index of the joint name in the joint trajectory message.
                    if joint_name in msg.joint_names:
                        joint_index = msg.joint_names.index(joint_name)
                        time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                        row = [time, msg.joint_names[joint_index], msg.points[0].positions[joint_index], msg.points[0].velocities[joint_index], msg.points[0].effort[joint_index]]
                        dfs_traj[list_joints.index(joint_name)] = pd.concat(
                            [dfs_traj[list_joints.index(joint_name)], pd.DataFrame([row], columns=header_df_traj)],
                            ignore_index=True
                        )

            if topic == '/joint_states':
                msg = deserialize_message(msg, JointState)
                for joint_name in list_joints:
                    # Get the index of the joint name in the joint state message.
                    if joint_name in msg.name:
                        joint_index = msg.name.index(joint_name)
                        time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                        row = [time, msg.name[joint_index], msg.position[joint_index], msg.velocity[joint_index], msg.effort[joint_index]]
                        dfs_state[list_joints.index(joint_name)] = pd.concat(
                            [dfs_state[list_joints.index(joint_name)], pd.DataFrame([row], columns=header_df_state)],
                            ignore_index=True
                        )

            counter += 1
            pbar.update(1)

    print("Finished processing messages")
    print(f"Number of messages processed: {counter}")

    # Save the csvs in the same directory as the rosbag.
    for i, df in enumerate(dfs_state):
        df.to_csv(rosbag_path + f'/{list_joints[i]}_state.csv', index=False)

    for i, df in enumerate(dfs_traj):
        df.to_csv(rosbag_path + f'/{list_joints[i]}_traj.csv', index=False)

# Load the csvs.
dfs_state = []
dfs_traj = []

DES_TIME_CUT = [40.0, 60.0]

for joint_name in list_joints:
    df_state = pd.read_csv(rosbag_path + f'/{joint_name}_state.csv')
    df_traj = pd.read_csv(rosbag_path + f'/{joint_name}_traj.csv')
    df_state['timestamp'] = df_state['timestamp'] - df_state['timestamp'][0]
    df_traj['timestamp'] = df_traj['timestamp'] - df_traj['timestamp'][0]

    df_state_filtered = df_state[(df_state['timestamp'] >= DES_TIME_CUT[0]) & (df_state['timestamp'] <= DES_TIME_CUT[1])]
    df_traj_filtered = df_traj[(df_traj['timestamp'] >= DES_TIME_CUT[0]) & (df_traj['timestamp'] <= DES_TIME_CUT[1])]
    dfs_state.append(df_state_filtered)
    dfs_traj.append(df_traj_filtered)

# Plot the joint positions.
fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
fig.suptitle('Joint Positions')
for i in range(8):
    col = 0 if i < 4 else 1
    row = i if i < 4 else i - 4
    ax[row, col].plot(dfs_state[i]['timestamp'], dfs_state[i]['pos'], label='pos_' + list_joints[i])
    ax[row, col].plot(dfs_traj[i]['timestamp'], dfs_traj[i]['pos_des'], label='pos_des_' + list_joints[i])
    ax[row, col].legend()
plt.tight_layout()

# Plot the joint velocities.
fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
fig.suptitle('Joint Velocities')
for i in range(8):
    col = 0 if i < 4 else 1
    row = i if i < 4 else i - 4
    ax[row, col].plot(dfs_state[i]['timestamp'], dfs_state[i]['vel'], label='vel_' + list_joints[i])
    ax[row, col].plot(dfs_traj[i]['timestamp'], dfs_traj[i]['vel_des'], label='vel_des_' + list_joints[i])
    ax[row, col].legend()
plt.tight_layout()

# Plot the joint efforts.
fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
fig.suptitle('Joint Effort')
for i in range(8):
    col = 0 if i < 4 else 1
    row = i if i < 4 else i - 4
    ax[row, col].plot(dfs_state[i]['timestamp'], dfs_state[i]['eff'], label='eff_' + list_joints[i])
    ax[row, col].plot(dfs_traj[i]['timestamp'], dfs_traj[i]['eff_des'], label='eff_des_' + list_joints[i])
    ax[row, col].legend()
plt.tight_layout()

plt.show()




