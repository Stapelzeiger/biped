import os
import gym
import torch
from gym.wrappers import RecordVideo
from sim_mujoco.imit_learning_setup.DAgger.nnpolicy import NNPolicy

"""File to evaluate (visualize) the DAgger policy for pendulum swing-up
"""

net_arch = [(3, 56), (56, 112), (112, 56), (56, 1)]
env = gym.make("Pendulum-v1")
wrapped_env = RecordVideo(env, os.path.join(os.path.dirname(os.path.realpath(__file__))))
policy = NNPolicy(net_arch)
policy.model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_dagger_policy.pt")))
obs = wrapped_env.reset()
done = False
while not done:
    u = 2 * policy.predict(obs).detach().numpy()
    res = wrapped_env.step(u)
    wrapped_env.render()
    obs = res[0]
    done = res[2]