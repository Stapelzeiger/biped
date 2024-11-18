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
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Vector3Stamped


# Parse the arguments.
parser = argparse.ArgumentParser(description="processes ROS 2 Bag directory path.")
parser.add_argument('--path', type=str, help="ROS 2 Bag directory path")
parser.add_argument('--extract_csvs', action='store_true', help="Extracts the messages to CSV files")
parser.add_argument('--plot_data', action='store_true', help="Plots the data")
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
    header_df_joint_state = 'timestamp', 'name', 'pos', 'vel', 'eff'
    df_joint_state = pd.DataFrame(columns=header_df_joint_state)
    dfs_joint_state = [df_joint_state for _ in range(len(list_joints))]

    # Create a dataframe for each joint trajectory.
    header_df_joint_traj = 'timestamp', 'name', 'pos_des', 'vel_des', 'eff_des'
    df_joint_traj = pd.DataFrame(columns=header_df_joint_traj)
    dfs_joint_traj = [df_joint_traj for _ in range(len(list_joints))]

    # Create a dataframe for the capture point DCM.
    header_df_dcm_des = 'timestamp', 'desired_dcm_x', 'desired_dcm_y',
    header_df_dcm_next = 'timestamp', 'next_footstep_x', 'next_footstep_y'
    header_df_dcm_predicted = 'timestamp', 'predicted_dcm_x', 'predicted_dcm_y'
    header_df_dcm = 'timestamp', 'dcm_x', 'dcm_y'
    df_dcm_desired = pd.DataFrame(columns=header_df_dcm_des)
    df_dcm = pd.DataFrame(columns=header_df_dcm)
    df_dcm_next = pd.DataFrame(columns=header_df_dcm_next)
    df_dcm_predicted = pd.DataFrame(columns=header_df_dcm_predicted)

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
                        dfs_joint_traj[list_joints.index(joint_name)] = pd.concat(
                            [dfs_joint_traj[list_joints.index(joint_name)], pd.DataFrame([row], columns=header_df_joint_traj)],
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
                        dfs_joint_state[list_joints.index(joint_name)] = pd.concat(
                            [dfs_joint_state[list_joints.index(joint_name)], pd.DataFrame([row], columns=header_df_joint_state)],
                            ignore_index=True
                        )

            if topic == '/capture_point/markers_next_footstep':
                msg = deserialize_message(msg, Marker)
                time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                row = [time, msg.pose.position.x, msg.pose.position.y]
                df_dcm_next = pd.concat([df_dcm_next, pd.DataFrame([row], columns=header_df_dcm_next)], ignore_index=True)

            if topic == '/capture_point/markers_dcm':
                msg = deserialize_message(msg, Marker)
                time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                row = [time, msg.pose.position.x, msg.pose.position.y]
                df_dcm = pd.concat([df_dcm, pd.DataFrame([row], columns=header_df_dcm)], ignore_index=True)

            if topic == '/capture_point/desired_dcm':
                msg = deserialize_message(msg, Vector3Stamped)
                time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                row_data = [time, msg.vector.x, msg.vector.y]
                df_dcm_desired = pd.concat([df_dcm_desired, pd.DataFrame([row_data], columns=header_df_dcm)], ignore_index=True)

            if topic == '/capture_point/predicted_dcm':
                msg = deserialize_message(msg, Vector3Stamped)
                time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                row_data = [time, msg.vector.x, msg.vector.y]
                df_dcm_predicted = pd.concat([df_dcm_predicted, pd.DataFrame([row_data], columns=header_df_dcm_predicted)], ignore_index=True)

            counter += 1

            pbar.update(1)

    print("Finished processing messages")
    print(f"Number of messages processed: {counter}")

    # Save the csvs in the same directory as the rosbag.
    for i, df in enumerate(dfs_joint_state):
        df.to_csv(rosbag_path + f'/{list_joints[i]}_state.csv', index=False)

    for i, df in enumerate(dfs_joint_traj):
        df.to_csv(rosbag_path + f'/{list_joints[i]}_traj.csv', index=False)

    df_dcm_desired.to_csv(rosbag_path + '/dcm_des.csv', index=False)
    df_dcm_next.to_csv(rosbag_path + '/dcm_next.csv', index=False)
    df_dcm_predicted.to_csv(rosbag_path + '/dcm_predicted.csv', index=False)
    df_dcm.to_csv(rosbag_path + '/dcm.csv', index=False)


def plot_data(rosbag_path):
    # Load the csvs.
    dfs_joint_state = []
    dfs_joint_traj = []

    DES_TIME_CUT = None

    for joint_name in list_joints:
        df_joint_state = pd.read_csv(rosbag_path + f'/{joint_name}_state.csv')
        df_joint_traj = pd.read_csv(rosbag_path + f'/{joint_name}_traj.csv')
        # df_joint_state['timestamp'] = df_joint_state['timestamp'] - df_joint_state['timestamp'][0]
        # df_joint_traj['timestamp'] = df_joint_traj['timestamp'] - df_joint_traj['timestamp'][0]

        if DES_TIME_CUT is not None:
            df_joint_state = df_joint_state[(df_joint_state['timestamp'] >= DES_TIME_CUT[0]) & (df_joint_state['timestamp'] <= DES_TIME_CUT[1])]
            df_joint_traj = df_joint_traj[(df_joint_traj['timestamp'] >= DES_TIME_CUT[0]) & (df_joint_traj['timestamp'] <= DES_TIME_CUT[1])]
        dfs_joint_state.append(df_joint_state)
        dfs_joint_traj.append(df_joint_traj)


    df_dcm_desired = pd.read_csv(rosbag_path + '/dcm_des.csv')
    df_dcm_desired['timestamp'] = df_dcm_desired['timestamp'] - df_dcm_desired['timestamp'][0]
    df_dcm = pd.read_csv(rosbag_path + '/dcm.csv')
    df_dcm['timestamp'] = df_dcm['timestamp'] - df_dcm['timestamp'][0]
    df_dcm_next = pd.read_csv(rosbag_path + '/dcm_next.csv')
    df_dcm_next['timestamp'] = df_dcm_next['timestamp'] - df_dcm_next['timestamp'][0]
    df_dcm_predicted = pd.read_csv(rosbag_path + '/dcm_predicted.csv')
    df_dcm_predicted['timestamp'] = df_dcm_predicted['timestamp'] - df_dcm_predicted['timestamp'][0]

    # Plot the joint positions.
    fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    fig.suptitle('Joint Positions' + rosbag_path)
    for i in range(8):
        col = 0 if i < 4 else 1
        row = i if i < 4 else i - 4
        ax[row, col].plot(dfs_joint_state[i]['timestamp'], dfs_joint_state[i]['pos'], label='pos_' + list_joints[i])
        ax[row, col].plot(dfs_joint_traj[i]['timestamp'], dfs_joint_traj[i]['pos_des'], label='pos_des_' + list_joints[i])
        ax[row, col].legend()
    plt.tight_layout()

    # Plot the joint velocities.
    fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    fig.suptitle('Joint Velocities ' + rosbag_path)
    for i in range(8):
        col = 0 if i < 4 else 1
        row = i if i < 4 else i - 4
        ax[row, col].plot(dfs_joint_state[i]['timestamp'], dfs_joint_state[i]['vel'], label='vel_' + list_joints[i])
        ax[row, col].plot(dfs_joint_traj[i]['timestamp'], dfs_joint_traj[i]['vel_des'], label='vel_des_' + list_joints[i])
        ax[row, col].legend()
    plt.tight_layout()

    # Plot the joint efforts.
    fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    fig.suptitle('Joint Effort' + rosbag_path)
    for i in range(8):
        col = 0 if i < 4 else 1
        row = i if i < 4 else i - 4
        ax[row, col].plot(dfs_joint_state[i]['timestamp'], dfs_joint_state[i]['eff'], label='eff_' + list_joints[i])
        ax[row, col].plot(dfs_joint_traj[i]['timestamp'], dfs_joint_traj[i]['eff_des'], label='eff_des_' + list_joints[i])
        ax[row, col].legend()
    plt.tight_layout()

    # Plot capture point data
    print(df_dcm_desired.head())
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle('Capture Point Data' + rosbag_path)
    # ax.plot(df_dcm_next['timestamp'], df_dcm_next['next_footstep_x'], '*', label='next_footstep')
    ax.plot(df_dcm_desired['timestamp'], df_dcm_desired['dcm_x'], 'go', label='desired_dcm')
    ax.plot(df_dcm['timestamp'], df_dcm['dcm_x'], 'r*', label='dcm')


if args.plot_data:

    # rosbag_path_old = '/home/sorina/Documents/code/biped_hardware/bags/20241013-11-33-35.bag'
    # plot_data(rosbag_path_old)

    plot_data(rosbag_path)

    plt.show()




