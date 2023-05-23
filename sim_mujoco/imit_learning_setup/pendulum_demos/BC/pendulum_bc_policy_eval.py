import os
import gym
import torch
from gym.wrappers import RecordVideo
from sim_mujoco.imit_learning_setup.behavioral_cloning.bc import BC_Agent

"""File to evaluate (visualize) the DAgger policy for pendulum swing-up
"""

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
wrapped_env = RecordVideo(env, os.path.dirname(os.path.realpath(__file__)))
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("device: ", device)

# Remember means/stds will be loaded, but need to pass tensors of the right shape
bc_agent = BC_Agent(policy_arch, torch.zeros(3), torch.zeros(3), torch.zeros(1), torch.zeros(1))
bc_agent.load_policy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_bc_policy.pt"))
bc_agent.to(device)

obs = wrapped_env.reset()
done = False
while not done:
    u = bc_agent.get_action(torch.tensor(obs, dtype=torch.float32).to(device)).detach().numpy()
    res = wrapped_env.step(u)
    wrapped_env.render()
    obs = res[0]
    done = res[2]