import numpy as np
from sim_mujoco.sim_mujoco_learning.mujoco_imit_node import MujocoImitNode
from sim_mujoco.sim_mujoco_learning.mujoco_imit_node_parallel import MujocoImitNodeParallel
from sim_mujoco.sim_mujoco_learning.expert_traj import ExpertTrajectory

class ILWrapper:

    def __init__(self, mujoco_xml_path, roll_len):
        self.mujoco_node = MujocoImitNode(mujoco_xml_path)
        self.roll_len = roll_len
        self.expert_traj = None
        self.action_shape = (len(ExpertTrajectory.ACTION_COL),)
        self.obs_shape = (len(ExpertTrajectory.OBS_COL),)

    def reset(self, exp_traj):
        self.expert_traj = exp_traj
        qpos0, qvel0 = exp_traj[0]['qpos'], exp_traj[0]['qvel']
        self.mujoco_node.reset(qpos0, qvel0)
        obs = ILWrapper._construct_observation(qpos0, qvel0)
        return obs
    
    def step(self, action):
        """Steps the environment forward one timestep with the given action

        Args:
            action (array-like): desired joint positions, desired joint velocities, and feedforward torque

        Returns:
            tuple: The resulting observation, reward, done_flag from environment
        """
        t, qpos, qvel = self.mujoco_node.step(action)
        
        # Construct observation from qpos, qvel
        obs = ILWrapper._construct_observation(qpos, qvel)
        # Compute reward from qpos, qvel
        reward = self.compute_reward(t, qpos, qvel)
        # Determine whether sim is done
        done = self.is_done(t, qpos, qvel)

        return obs, reward, done
    
    staticmethod
    def _construct_observation(qpos, qvel):
        # TODO: Construct the observation from qpos, qvel
        return [qpos, qvel]

    def compute_reward(self, t, qpos, qvel):
        """Computes the reward associated with being in a given state at a given time

        Args:
            t (float): Time at which to determine reward
            qpos (array-like): joint positions
            qvel (array-like): joint velocities

        Returns:
            float: Reward associated with being in the given state at the given time
        """
        # TODO: implement reward function
        return 0

    def is_done(self, t, qpos, qvel):
        """Computes whether the rollout is done

        Args:
            t (float): Time at which to determine if rollout is done
            qpos (array-like): joint positions
            qvel (array-like): joint velocities

        Returns:
            float: Reward associated with being in the given state at the given time
        """
        return False


class VecILWrapper:

    def __init__(self, num_envs, mujoco_xml_path, roll_len):
        self.num_envs = num_envs
        self.roll_len = roll_len
        self.mujoco_nodes = MujocoImitNodeParallel(mujoco_xml_path, self.num_envs)
        self.expert_trajs = None
        self.single_action_shape = (len(ExpertTrajectory.ACTION_COL),)
        self.single_obs_shape = (len(ExpertTrajectory.OBS_COL),)

    def reset(self, exp_trajs):
        self.expert_trajs = exp_trajs
        qpos0 = np.vstack((exp_trajs[ii].get_qpos_step(0) for ii in range(self.num_envs)))
        qvel0 = np.vstack((exp_trajs[ii].get_qvel_step(0) for ii in range(self.num_envs)))
        self.mujoco_nodes.reset(qpos0, qvel0)
        obs = VecILWrapper._construct_observation(qpos0, qvel0)
        return obs
    
    def step(self, action):
        """Steps the environment forward one timestep with the given action

        Args:
            action (array-like): desired joint positions, desired joint velocities, and feedforward torque

        Returns:
            tuple: The resulting observation, reward, done_flag from environment
        """
        
        t, qpos, qvel = self.mujoco_nodes.step(action)
            
        # Construct observation from qpos, qvel
        obs = VecILWrapper._construct_observation(qpos, qvel)
        # Compute reward from qpos, qvel
        reward = self.compute_reward(t, qpos, qvel)
        # Determine whether sim is done
        done = self.is_done(t, qpos, qvel)

        return obs, reward, done
    
        staticmethod
    def _construct_observation(qpos, qvel):
        # TODO: Construct the observation from qpos, qvel
        return [qpos, qvel]

    def compute_reward(self, t, qpos, qvel):
        """Computes the reward associated with being in a given state at a given time

        Args:
            t (float): Time at which to determine reward
            qpos (array-like): joint positions
            qvel (array-like): joint velocities

        Returns:
            float: Reward associated with being in the given state at the given time
        """
        # TODO: implement reward function
        return 0

    def is_done(self, t, qpos, qvel):
        """Computes whether the rollout is done

        Args:
            t (float): Time at which to determine if rollout is done
            qpos (array-like): joint positions
            qvel (array-like): joint velocities

        Returns:
            float: Reward associated with being in the given state at the given time
        """
        return False