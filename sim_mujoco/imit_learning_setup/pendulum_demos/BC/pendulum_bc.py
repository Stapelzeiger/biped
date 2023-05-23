from sim_mujoco.imit_learning_setup.pendulum_demos.pendulum_expert import pendulum_expert
from sim_mujoco.imit_learning_setup.behavioral_cloning.bc import bc
import gym
import numpy as np
import torch
import os
"""File for training a policy using DAgger to solve pendulum swing-up

Note: had to modify /home/wcompton/anaconda3/envs/imit_biped/lib/python3.10/site-packages/gym/envs/classic_control/pendulum.py
change line 44 to
        newthdot = thdot + (1 * g / (2 * l) * np.sin(th) + 1.0 / (m * l ** 2) * u) * dt
otherwise physics was broken
"""

print(os.path.dirname(os.path.realpath(__file__)))
policy_arch = [
    {'Layer': 'Linear', 'Input': 3, 'Output': 56, 'SpectralNorm': False},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 112, 'SpectralNorm': False},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 112, 'Output': 56, 'SpectralNorm': False},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 1, 'SpectralNorm': False}
]
env = gym.make("Pendulum-v1")
epochs = 100
num_traj = 50

states = np.zeros((200 * num_traj, 3))
actions = np.zeros((200 * num_traj, 1))
for ii in range(num_traj):
    obs = env.reset()
    done = False
    jj = 0
    while not done:
        states[ii * 200 + jj, :] = obs
        u = pendulum_expert(obs)
        actions[ii * 200 + jj] = u[0]
        res = env.step(u)
        obs = res[0]
        done = res[2]
        jj += 1


bc_trainer = bc(states, actions, policy_arch)

bc_trainer.train(epochs)
bc_trainer.save_policy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_bc_policy.pt"))
