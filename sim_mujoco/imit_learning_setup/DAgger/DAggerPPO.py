import numpy as np
import random
from stable_baselines3.ppo.ppo_imit import PPOImit
from stable_baselines3.common.buffers import RolloutBufferMod


class DAggerPPO:

    def __init__(self, env, expert, policy, beta_sched, Tmax, stopping_cond, N):
        """ Notes
        The 'env', and 'expert' are critical parts here. 'env' should act like a gym environment.
        I.E. it should have a reset() command which places it into some initial state, as well as 
        a 'step(action)' function which move sim forward, and returns the observation and a done
        flag (done being equivalent to falling over prematurely for walking)
        The 'expert' simply needs (or rather is, in this implementation) a function which takes in
        the current observation and outputs the expert action.
        """
        self.env = env                      # Environment to collect data in
        self.expert = expert                # Expert policy
        self.rollout_iters = 0              # Number of iterations
        self.beta_schedule = beta_sched     # Schedule for expert proportion
        self.Tmax = Tmax                    # Maximum steps per trajectory
        self.stopping_cond = stopping_cond  # Conditions on state to stop trajectory
        self.N = N                          # Number of trajectories to collect
        self.policy = policy                # policy network (PPO)

    def dagger_rollout(self) -> tuple:
        print(f"Collecting Rollouts {self.rollout_iters}")
        # Initialize memory for states and expert actions
        state_memory = np.zeros((self.Tmax * self.N, self.env.observation_space.shape[0]))
        expert_action_memory = np.zeros((self.Tmax * self.N, self.env.action_space.shape[0]))
        # Grab the beta weighting expert vs. policy
        beta = self.beta_schedule[self.rollout_iters]

        # Run a Dagger Rollout
        sample_ind = 0
        for _ in range(self.N):
            # Begin a trajectory
            traj_steps = 0
            traj_over = False
            obs = self.env.reset() # Reset environment to random IC at beginning
            obs = obs[0] # SJ: IDK what gym version we should use, but under 0.26.2, obs looks like 
            #    (array([cosT , sinT,  omega ], dtype=float32), {}), so it throws index-out-of-range error without obs = obs[0].
            while not traj_over and traj_steps < self.Tmax:
                # Get the expert action at the current state
                expert_action = self.expert(obs)

                # Record the expert action and state in the memory
                state_memory[sample_ind, :] = obs
                expert_action_memory[sample_ind, :] = expert_action

                # Execute either the expert action or policy action in the environment
                if random.random() < beta:
                    result = self.env.step(expert_action)
                else:
                    policy_action = self.policy.predict(obs)
                    result = self.env.step(policy_action[0]) # another difference in PPO: type of obj policy_action is returned as
                # Grab the observation returned from the step
                obs = result[0]

                # Check if the trajectory reached a termination condition 
                if result[2]:
                    traj_over = True
                
                # Increment counters
                traj_steps += 1
                sample_ind += 1
        self.rollout_iters += 1
        # Return the DAgger Rollout
        return (state_memory[:sample_ind, :], expert_action_memory[:sample_ind, :])
    
    
    def train_dagger(self, epochs):
        """train_dagger runs pure behavior cloning on a deterministic policy. 
        The algorithm fits a neural network to the collected data, copying actions of the 
        expert policy

        Args:
            epochs (int): number of epochs to train
        """
        train_states = np.zeros((0, self.env.observation_space.shape[0]))
        train_actions = np.zeros((0, self.env.action_space.shape[0]))
        # Run for a specific number of iterations
        for _ in range(epochs):
            # Perform a DAgger rollout
            states, actions = self.dagger_rollout()
            # Append results to the set of data to train behavior cloning
            train_states = np.vstack((train_states, states))
            train_actions = np.vstack((train_actions, actions))

            print("train_states shape: ", train_states.shape)
            print("train_actions shape: ", train_actions.shape)
            
            # Train the model (implementation detail: in contrast to DAgger + NNPolicy, 
            # DAgger + PPO collects all rollouts together and trains at once, not loop-by-loop)
            rollout_buffer = RolloutBufferMod(
                buffer_size=self.Tmax, 
                observations=train_states, 
                actions=train_actions
                )

            self.policy.train(rollout_buffer)


#######################################################################################


# class BaseBuffer(ABC):
#     """
#     Base class that represent a buffer (rollout or replay)

#     :param buffer_size: Max number of element in the buffer
#     :param device: PyTorch device
#         to which the values will be converted
#     """

#     def __init__(
#         self,
#         buffer_size: int,
#         device: Union[th.device, str] = "auto",
#     ):
#         super().__init__()
#         self.buffer_size = buffer_size
#         self.pos = 0
#         self.full = False
#         self.device = get_device(device)

#     @staticmethod
#     def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
#         """
#         Swap and then flatten axes 0 (buffer_size) and 1
#         to convert shape from [n_steps, ...] (when ... is the shape of the features)
#         to [n_steps, ...] (which maintain the order)

#         :param arr:
#         :return:
#         """
#         shape = arr.shape
#         if len(shape) < 3:
#             shape = (*shape, 1)
#         return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

#     def size(self) -> int:
#         """
#         :return: The current size of the buffer
#         """
#         if self.full:
#             return self.buffer_size
#         return self.pos

#     # def add(self, *args, **kwargs) -> None:
#     #     """
#     #     Add elements to the buffer.
#     #     """
#     #     raise NotImplementedError()

#     def extend(self, *args, **kwargs) -> None:
#         """
#         Add a new batch of transitions to the buffer
#         """
#         # Do a for loop along the batch axis
#         for data in zip(*args):
#             self.add(*data)

#     # def reset(self) -> None:
#     #     """
#     #     Reset the buffer.
#     #     """
#     #     self.pos = 0
#     #     self.full = False

#     def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
#         """
#         :param batch_size: Number of element to sample
#         :param env: associated gym VecEnv
#             to normalize the observations/rewards when sampling
#         :return:
#         """
#         upper_bound = self.buffer_size if self.full else self.pos
#         batch_inds = np.random.randint(0, upper_bound, size=batch_size)
#         return self._get_samples(batch_inds, env=env)

#     @abstractmethod
#     def _get_samples(
#         self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
#     ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
#         """
#         :param batch_inds:
#         :param env:
#         :return:
#         """
#         raise NotImplementedError()

#     def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
#         """
#         Convert a numpy array to a PyTorch tensor.
#         Note: it copies the data by default

#         :param array:
#         :param copy: Whether to copy or not the data (may be useful to avoid changing things
#             by reference). This argument is inoperative if the device is not the CPU.
#         :return:
#         """
#         if copy:
#             return th.tensor(array, device=self.device)
#         return th.as_tensor(array, device=self.device)

#     @staticmethod
#     def _normalize_obs(
#         obs: Union[np.ndarray, Dict[str, np.ndarray]],
#         env: Optional[VecNormalize] = None,
#     ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
#         if env is not None:
#             return env.normalize_obs(obs)
#         return obs

#     @staticmethod
#     def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
#         if env is not None:
#             return env.normalize_reward(reward).astype(np.float32)
#         return reward



