import numpy as np
import random
import os
import matplotlib.pyplot as plt
from expert_ackermann import states_2_control_states

def wrap_circular_value(input_value):
    return (input_value + np.pi) % (2*np.pi) - np.pi

class DAgger:

    def __init__(self, env, expert, policy, desired_trajectory, beta_sched, Tmax, stopping_cond, N, input_NN_size, seed=1234):
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
        self.policy = policy                # policy network
        self.dt = self.env.dt               # Time step of the environment
        self.desired_trajectory = desired_trajectory # Fcn des trajectory to follow
        self.input_NN_size = input_NN_size  # Size of the input to the NN
        self.rng = np.random.default_rng(seed=seed)

        self.save_folder_plots = "plots"    # Folder to save plots
        # if the folder exists, delete it
        if os.path.exists(self.save_folder_plots):
            os.system(f"rm -rf {self.save_folder_plots}")
        os.makedirs(self.save_folder_plots) # Create the folder

    def dagger_rollout(self) -> tuple:
        print(f"Collecting Rollouts {self.rollout_iters}")
        # Initialize memory for states and expert actions
        state_memory = np.zeros((self.Tmax * self.N, self.input_NN_size))

        expert_action_memory = np.zeros((self.Tmax * self.N, self.env.udim))
        # Grab the beta weighting expert vs. policy
        beta = self.beta_schedule[self.rollout_iters]

        # Run a Dagger Rollout
        sample_ind = 0
        for _ in range(self.N):

            # Create a plot for the trajectory
            traj_list = []
            des_traj_list = []

            # Begin a trajectory
            traj_steps = 0

            x_d = self.desired_trajectory(self.Tmax, self.env.dt)

            self.env.state = x_d[0, 0:self.env.xdim].copy() # Set the initial state to the first desired state

            obs_k = self.env.state.copy()
            obs_k_prev = obs_k.copy()

            reward_total = 0
            reward_list = []
            pos_B_error_list = []
            policy_action_list = []

            while traj_steps < self.Tmax:
                # Get the expert action at the current state

                # This is for the ackermann pb, so we can change everything in the body frame
                x, x_d_modified = states_2_control_states(obs_k_prev, obs_k, x_d[traj_steps, :], self.env.dt)
                obs_k_prev = obs_k.copy()
                expert_action = self.expert(x, x_d_modified)
                state_memory[sample_ind, :] = np.concatenate((x, x_d_modified))

                expert_action_memory[sample_ind, :] = expert_action

                # Execute either the expert action or policy action in the environment
                # if self.rollout_iters == 0: # indicator function, as reported in the paper, performs best
                if random.random() < beta:
                    obs_k, reward, _, _ = self.env.step(expert_action, x_d[traj_steps, :])
                else:
                    input_NN = np.concatenate((x, x_d_modified))
                    policy_action = self.policy.forward(input_NN)
                    obs_k, reward, _, _ = self.env.step(policy_action.detach().numpy())
                    policy_action_list.append(policy_action.detach().numpy())

                # Append to list
                traj_list.append(np.array([obs_k[0], obs_k[1], x[2], x[3], x[4], x[5]])) # first two are in I frame, the rest in B frame
                des_traj_list.append(x_d[traj_steps, :])
                pos_B_error_list.append(x_d_modified[0:2])

                # Increment counters
                traj_steps += 1
                sample_ind += 1

                # Reward.
                reward_list.append(reward_total)
                reward_total += reward

            _, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(np.array(traj_list)[:, 0], np.array(traj_list)[:, 1], label='Trajectory')
            ax.plot(np.array(des_traj_list)[:, 0], np.array(des_traj_list)[:, 1], label='Desired Trajectory')
            ax.legend()
            plt.savefig(f"{self.save_folder_plots}/traj_rollout_{self.rollout_iters}.png")

            # if self.rollout_iters > 0:
            # policy_action_list = np.array(policy_action_list)
            # _, ax = plt.subplots(2, 1, figsize=(5, 5))
            # ax[0].plot(policy_action_list[:, 0], label='u_v')
            # ax[1].plot(policy_action_list[:, 1], label='u_steering')
            # plt.legend()
            # plt.savefig(f"{self.save_folder_plots}/policy_{self.rollout_iters}.png")


        self.rollout_iters += 1
        # Return the DAgger Rollout
        return (state_memory[:sample_ind, :], expert_action_memory[:sample_ind, :])

    def train_dagger(self, epochs):
        """train_dagger runs pure behaviour cloning on a deterministic policy. 
        The algorithm fits a neural network to the collected data, copying actions of the 
        expert policy

        Args:
            epochs (int): number of epochs to train
        """
        train_states = np.zeros((0, self.input_NN_size))
        train_actions = np.zeros((0, self.env.udim))
        best_mse_list = []
        # Run for a specific number of iterations
        for _ in range(epochs):
            # Perform a DAgger rollout
            states, expert_actions = self.dagger_rollout()
            # Append results to the set of data to train behaviour cloning
            train_states = np.vstack((train_states, states))
            train_actions = np.vstack((train_actions, expert_actions))

            # Train the model
            best_mse, history_mse = self.policy.train(train_states, train_actions)
            best_mse_list.append(best_mse)

            _, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(history_mse)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("MSE")
            ax.set_title("MSE vs. Epochs")
            plt.savefig(f"{self.save_folder_plots}/MSE_vs_Epochs_{self.rollout_iters}.png")

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(best_mse_list)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE")
        ax.set_title("MSE vs. Epochs")
        plt.savefig(f"{self.save_folder_plots}/MSE_vs_Epochs.png")
