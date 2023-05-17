import os
import numpy as np
import pandas as pd
import gym
import pybullet_envs

from ppo import PPO

policy_name = "ppo_policy_6"

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
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
    [make_env("HalfCheetahBulletEnv-v0", 1 + i, i, False, policy_name) for i in range(1)]
)

policy_arch = [
    {'Layer': 'Linear', 'Input': np.array(envs.single_observation_space.shape).prod(), 'Output': 64, 'std': np.sqrt(2)},
    {'Layer': 'Tanh'},
    {'Layer': 'Linear', 'Input': 64, 'Output': 64, 'std': np.sqrt(2)},
    {'Layer': 'Tanh'},
    {'Layer': 'Linear', 'Input': 64, 'Output': np.array(envs.single_action_space.shape).prod(), 'std': 0.01},
]
value_arch = [
    {'Layer': 'Linear', 'Input': np.array(envs.single_observation_space.shape).prod(), 'Output': 64, 'std': np.sqrt(2)},
    {'Layer': 'Tanh'},
    {'Layer': 'Linear', 'Input': 64, 'Output': 64, 'std': np.sqrt(2)},
    {'Layer': 'Tanh'},
    {'Layer': 'Linear', 'Input': 64, 'Output': 1, 'std': 1.0},
]
train_epochs = 1000 


def main():

    # Run vanilla behavior cloning
    behavior_clone_ppo = PPO(envs, policy_arch, value_arch, rng_seed=1)
    behavior_clone_ppo.train(train_epochs)

    behavior_clone_ppo.save_policy(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"policies/{policy_name}.pt"))
    

if __name__ == "__main__":
    main()