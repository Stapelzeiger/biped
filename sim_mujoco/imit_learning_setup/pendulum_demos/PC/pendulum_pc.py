from sim_mujoco.imit_learning_setup.pendulum_demos.pendulum_expert import pendulum_expert
from sim_mujoco.imit_learning_setup.pendulum_demos.PC.pc import ILWrapper
import gym
import numpy as np
import torch
import os

from sim_mujoco.imit_learning_setup.AnimalImit.ppo import PPO

"""File for training a policy using DAgger to solve pendulum swing-up

Note: had to modify /home/wcompton/anaconda3/envs/imit_biped/lib/python3.10/site-packages/gym/envs/classic_control/pendulum.py
change line 44 to
        newthdot = thdot + (1 * g / (2 * l) * np.sin(th) + 1.0 / (m * l ** 2) * u) * dt
otherwise physics was broken
"""

print(os.path.dirname(os.path.realpath(__file__)))

# policy_arch = [
#     {'Layer': 'Linear', 'Input': 3, 'Output': 56, 'std': np.sqrt(2)},
#     {'Layer': 'ReLU'},
#     {'Layer': 'Linear', 'Input': 56, 'Output': 56, 'std': np.sqrt(2)},
#     {'Layer': 'ReLU'},
#     {'Layer': 'Linear', 'Input': 56, 'Output': 1, 'std': 0.01},
# ]
# value_arch = [
#     {'Layer': 'Linear', 'Input': 3, 'Output': 56, 'std': np.sqrt(2)},
#     {'Layer': 'ReLU'},
#     {'Layer': 'Linear', 'Input': 56, 'Output': 56, 'std': np.sqrt(2)},
#     {'Layer': 'ReLU'},
#     {'Layer': 'Linear', 'Input': 56, 'Output': 1, 'std': 1.0},
# ]
policy_arch = [
    {'Layer': 'Linear', 'Input': 3, 'Output': 56},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 56},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 1},
]
value_arch = [
    {'Layer': 'Linear', 'Input': 3, 'Output': 56},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 56},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 1},
]
epochs = 200
num_traj = 1

env = gym.make("Pendulum-v1")
states = np.zeros((num_traj, 200, 3))
actions = np.zeros((num_traj, 200, 1))
for ii in range(num_traj):
    obs = env.reset()
    done = False
    jj = 0
    while not done:
        states[ii, jj, :] = obs
        u = pendulum_expert(obs)
        actions[ii, jj] = u[0]
        res = env.step(u)
        obs = res[0]
        done = res[2]
        jj += 1

# env_wrap = ILWrapper(states)
np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "exp_traj.npy"), states)
def make_env(seed, idx, capture_video, run_name):
    def thunk():
        env = ILWrapper(states, actions)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

envs = gym.vector.SyncVectorEnv(
    [make_env(1 + i, i, False, "pendulum_pc_policy") for i in range(1)]
)
pc_trainer = PPO(envs, policy_arch, value_arch)

pc_trainer.train(epochs)
pc_trainer.save_policy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_pc_policy.pt"))
