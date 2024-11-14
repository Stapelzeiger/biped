import gymnasium as gym
import new_robot_env
import torch
import numpy as np

from model import Policy, Value
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG


env = gym.make("gymnasium_env/BipedEnv", render_mode="human")
observation, info = env.reset()
device = wrap_env(env).device

episode_over = False

models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.action_space, device)

agent = PPO(models=models,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

agent.load('ppo/models/testing/24-11-13_00-40-00-071090_PPO/checkpoints/agent_5000.pt')

time = 0
while not episode_over:
    inputs = torch.from_numpy((observation.astype(np.float32)))
    action = agent.act(inputs, time, 200)  # agent policy that uses the observation and info
    action = agent.random_act(inputs)
    time += 1

    print("OBSERVATION----------------------")
    print(observation)
    print("ACTION---------------------------")
    print(action[0].detach().numpy())
    observation, reward, terminated, truncated, info = env.step(action[0].detach().numpy())
    # episode_over = terminated or truncated

env.close()