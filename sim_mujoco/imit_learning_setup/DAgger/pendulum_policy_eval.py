import gym
import torch
from nnpolicy import NNPolicy
import os
"""File to evaluate (visualize) the DAgger policy for pendulum swing-up
"""

net_arch = [(3, 56), (56, 112), (112, 56), (56, 1)]
env = gym.make("Pendulum-v1")
policy = NNPolicy(net_arch)
policy.model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_dagger_policy.pt")))
obs = env.reset()
done = False
while not done:
    u = 2 * policy.predict(obs).detach().numpy()
    res = env.step(u)
    env.render()
    obs = res[0]
    done = res[2]