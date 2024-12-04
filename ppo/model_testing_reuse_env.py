import gymnasium as gym
import new_robot_env
import torch
import numpy as np

from model import Policy, Value
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

#  xml_file="/home/zhonghezheng13579/ros_ws/src/biped/biped_robot_description/urdf/custom_robot.mujoco.xml
env = gym.make_vec('Walker2d-v5', render_mode="human", xml_file="/home/zhonghezheng13579/ros_ws/src/biped/biped_robot_description/urdf/custom_robot.mujoco.xml")

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
    action = agent.act(inputs, time, 100)  # agent policy that uses the observation and info
    action = np.ones(len(action[0][0]))
    time += 1

    observation, reward, terminated, truncated, info = env.step([action])

    # episode_over = terminated or truncated

env.close()