from pendulum_expert import pendulum_expert
from DAggerPPO import DAggerPPO
from stable_baselines3.ppo.ppo_imit import PPOImit
from nnpolicy import NNPolicy
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
# policy = NNPolicy(net_arch)
policy = PPOImit(
    policy="MlpPolicy", # one of MlpPolicy, CnnPolicy, MultiInputPolicy
    env=env
    )
epochs = 5

dagger_trainer = DAggerPPO(
    env, pendulum_expert, policy, np.linspace(1, 0, epochs), 200, None, 50
)

dagger_trainer.train_dagger(epochs)

torch.save(policy.model.state_dict(), os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_dagger_ppo_policy.pt"))
