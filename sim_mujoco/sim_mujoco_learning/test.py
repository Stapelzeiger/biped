import numpy as np
import pandas as pd
from time import sleep
import sys
sys.path.append("/home/leo/biped_ws/src/biped/sim_mujoco/")
from sim_mujoco_learning.mujoco_imit_node import MujocoImitNode
model_path = "/home/leo/biped_ws/src/biped/biped_robot_description/urdf/custom_robot.mujoco.xml"

data_path = "data/dataset_qpos.csv"
mj_node = MujocoImitNode(model_path, visualize=True, vis_rate=60)
dataset = pd.read_csv(data_path)

# Skip forward a bit in trajectory
dataset = dataset.iloc[0:, :]
print(dataset)
# Grab relevant variables
qpos = dataset[['q_0', 'q_1', 'q_2', 'q_3','q_4', 'q_5','q_6', 'q_7', 'q_8', 'q_9', 'q_10', 'q_11','q_12', 'q_13','q_14', 'q_15', 'q_16']].to_numpy()
qvel = dataset[['qd_0', 'qd_1', 'qd_2', 'qd_3','qd_4', 'qd_5','qd_6', 'qd_7', 'qd_8', 'qd_9', 'qd_10', 'qd_11','qd_12', 'qd_13','qd_14', 'qd_15']].to_numpy()
qfrc = dataset[['qfrc_applied_0', 'qfrc_applied_1', 'qfrc_applied_2', 'qfrc_applied_3', 'qfrc_applied_4', 'qfrc_applied_5', 'qfrc_applied_6', 
               'qfrc_applied_7', 'qfrc_applied_8', 'qfrc_applied_9', 'qfrc_applied_10', 'qfrc_applied_11', 'qfrc_applied_12', 'qfrc_applied_13', 
               'qfrc_applied_14', 'qfrc_applied_15'
]].to_numpy()
cntr = dataset[[
    'ctrl_0', 'ctrl_1', 'ctrl_2', 'ctrl_3', 'ctrl_4', 'ctrl_5', 'ctrl_6', 'ctrl_7', 'ctrl_8', 'ctrl_9', 'ctrl_10', 'ctrl_11', 
    'ctrl_12', 'ctrl_13', 'ctrl_14', 'ctrl_15', 'ctrl_16', 'ctrl_17', 'ctrl_18'
]].to_numpy()

# Format as action = [uff, qdes, qddes]
u_ff = np.hstack((qfrc[:, 6:10], qfrc[:, 11:15]))
q_des = np.hstack((cntr[:, :7:2], cntr[:, 10:17:2]))
qd_des = np.hstack((cntr[:, 1:8:2], cntr[:, 11:18:2]))
actions = np.hstack((u_ff, q_des, qd_des))

# Reset the env to the initial position
mj_node.reset(qpos[0, :], qvel[0, :])
mj_node.viewer.render()
# sleep(1)

print(actions)
# And sim forward
i = 1
while i < actions.shape[0]:
    print(i)
    action = actions[i, :]
    sleep(0.5)
    mj_node.step(action)
    i += 1
mj_node.close_writer()
