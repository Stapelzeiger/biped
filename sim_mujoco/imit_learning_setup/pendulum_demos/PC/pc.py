import gym
import numpy as np

class ILWrapper:

    def __init__(self, expert_trajs, exp_actions):
        self.env = gym.make("Pendulum-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.expert_trajs = expert_trajs
        self.expert_actions = exp_actions
        self.expert_traj = None
        self.action_shape = (1,)
        self.obs_shape = (3,)
        self.t = 0
        self.iterations = 0
        self.single_observation_space = np.array([1, 1, 8])
        self.single_action_space = np.array([2])
        self.metadata = self.env.metadata

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.expert_traj = self.expert_trajs[self.iterations % self.expert_trajs.shape[0], :, :]
        qpos0 = self.expert_traj[0, :2]
        qvel0 = self.expert_traj[0, 2]
        theta0 = np.arctan2(qpos0[0], qpos0[1])
        self.env.reset()
        self.env.state = np.array([theta0, qvel0])
        obs = np.hstack((qpos0, [qvel0]))
        self.t = 0
        self.reward_total = 0
        return obs
    
    def step(self, action):
        """Steps the environment forward one timestep with the given action

        Args:
            action (array-like): desired joint positions, desired joint velocities, and feedforward torque

        Returns:
            tuple: The resulting observation, reward, done_flag from environment
        """
        self.t += 1
        res = self.env.step(action)
        obs = res[0]
        
        # Compute reward from qpos, qvel
        reward = self.compute_reward(obs)
        self.reward_total += reward
        # Determine whether sim is done
        done = res[2]
        ret_dict = {}
        if done:
            ret_dict['episode'] = {}
            ret_dict['episode']['r'] = self.reward_total
            ret_dict['episode']['l'] = self.t
            self.reset()


        return obs, reward, done, ret_dict

    def compute_reward(self, obs):
        """Computes the reward associated with being in a given state at a given time

        Args:
            t (float): Time at which to determine reward
            qpos (array-like): joint positions
            qvel (array-like): joint velocities

        Returns:
            float: Reward associated with being in the given state at the given time
        """
        
        if self.t >= 200:
            e = obs - self.expert_traj[199, :]
        else:
            e = obs - self.expert_traj[self.t, :]
        e[0] *= 1
        return -np.linalg.norm(e)

    def close(self):
        self.env.close()


