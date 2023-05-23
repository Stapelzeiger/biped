import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from sim_mujoco.imit_learning_setup.AnimalImit.ppo import Agent
from sim_mujoco.imit_learning_setup.pendulum_demos.PC.pc import ILWrapper

"""File to evaluate (visualize) the DAgger policy for pendulum swing-up
"""

policy_arch = [
    {'Layer': 'Linear', 'Input': 3, 'Output': 56, 'std': np.sqrt(2)},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 56, 'std': np.sqrt(2)},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 1, 'std': 0.01},
]
value_arch = [
    {'Layer': 'Linear', 'Input': 3, 'Output': 56, 'std': np.sqrt(2)},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 56, 'std': np.sqrt(2)},
    {'Layer': 'ReLU'},
    {'Layer': 'Linear', 'Input': 56, 'Output': 1, 'std': 1.0},
]
states = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "exp_traj.npy"))
plt.plot(states[0, :, 2])
plt.show()
envs = ILWrapper(states)

# Remember means/stds will be loaded, but need to pass tensors of the right shape
agent = Agent(envs, policy_arch, value_arch)
agent.load_policy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_pc_policy.pt"))
agent.set_eval_mode(True)

obs = envs.reset()
done = False
with torch.no_grad():
    while not done:
        u = agent.get_action_and_value(torch.Tensor(obs))
        envs.env.render()
        res = envs.step(u.cpu().numpy())
        obs = res[0]
        done = res[2]