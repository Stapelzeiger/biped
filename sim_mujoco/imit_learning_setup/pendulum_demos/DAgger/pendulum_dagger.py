from sim_mujoco.imit_learning_setup.pendulum_demos.pendulum_expert import pendulum_expert
from sim_mujoco.imit_learning_setup.DAgger.DAgger import DAgger
from sim_mujoco.imit_learning_setup.DAgger.nnpolicy import NNPolicy
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
net_arch = [(3, 56), (56, 112), (112, 56), (56, 1)]
env = gym.make("Pendulum-v1")
epochs = 10
batch_size = 32
policy = NNPolicy(net_arch, epochs=epochs, batch_size=batch_size)
iterations = 5
traj_per_iter = 10


dagger_trainer = DAgger(
    env, pendulum_expert, policy, np.linspace(1, 0, iterations), 200, None, traj_per_iter
)

dagger_trainer.train_dagger(iterations)

torch.save(policy.model.state_dict(), os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_dagger_policy.pt"))
