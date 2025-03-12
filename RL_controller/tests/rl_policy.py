from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
from jax import numpy as jp
import os
from etils import epath
import sys

jax.config.update('jax_platform_name', 'cpu')

# Load policy.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
print("Available devices:", jax.devices())
print(str(jax.local_devices()[0]))

RESULTS_FOLDER_PATH ='/home/sorina/Documents/code/biped_hardware/ros2_ws/src/biped/RL_controller/results'

# Sort by date and get the latest folder.
folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
latest_folder = folders[-1]
print(f'Latest folder: {latest_folder}')

# In the latest folder, find the latest folder, ignore the files.
folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
print(folders)

folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]
if len(folders) == 0:
    raise ValueError(f'No folders found in {epath.Path(RESULTS_FOLDER_PATH) / latest_folder}')
    sys.exit()
if len(folders) > 1:
    latest_weights_folder = folders[-1]
else:
    latest_weights_folder = folders
print(f'Latest weights folder: {latest_weights_folder}')

path = epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder
print(f'Loading policy from: {path}')

policy_fn = ppo_checkpoint.load_policy(path)
jit_policy = jax.jit(policy_fn)

import time
rng = jax.random.PRNGKey(1)

frequencies = []

for i in range(1000):
    time_now = time.time()
    obs = {
    'privileged_state': jp.zeros(102),
    'state': jp.zeros(46),
    }
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_policy(obs, act_rng)
    DT = time.time() - time_now

    frequencies.append(1/DT)

import matplotlib
import matplotlib.pyplot as plt

# moving average
window = 10
frequencies = [sum(frequencies[i:i+window])/window for i in range(len(frequencies)-window)]
plt.plot(frequencies)

plt.ylim([500, 5000])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Call of the inference function')
plt.show()
