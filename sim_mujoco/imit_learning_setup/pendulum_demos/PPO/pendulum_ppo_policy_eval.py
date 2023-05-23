import os
import gym
import torch
import numpy as np
from sim_mujoco.imit_learning_setup.AnimalImit.ppo import Agent

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
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

envs = gym.vector.SyncVectorEnv(
    [make_env("Pendulum-v1", 1 + i, i, True, "pendulum_ppo_policy") for i in range(1)]
)

# Remember means/stds will be loaded, but need to pass tensors of the right shape
agent = Agent(envs, policy_arch, value_arch)
agent.load_policy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "pendulum_ppo_policy.pt"))
agent.set_eval_mode(True)

obs = envs.reset()
done = np.array([False, False])
with torch.no_grad():
    while not done.any():
        u = agent.get_action_and_value(torch.Tensor(obs))
        res = envs.step(u.cpu().numpy())
        obs = res[0]
        done = res[2]