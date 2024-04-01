import numpy as np
import random

# Util functions.
# def wrap_circular_value(input_value):
#     return (input_value + np.pi) % (2*np.pi) - np.pi

# def clamp(input_value, min_value, max_value):
#     if (input_value > max_value):
#         return max_value
#     if (input_value < min_value):
#         return min_value
#     return input_value

# # Traj fcns.
# def compute_fig8_simple(period, length, current_time, initial_state_I):
#     t = current_time
#     omega = 2 * np.pi / period
#     x = length * np.sin(omega * t)
#     y = length/2  * np.sin(2 * omega * t)
#     z = 0.0
#     vel_x = length * omega * np.cos(omega * t)
#     vel_y = length * omega * np.cos(2 * omega * t)
#     vel_z = 0.0

#     fig_8_start_heading = initial_state_I[2] - np.pi/4
#     R = np.array([[np.cos(fig_8_start_heading), -np.sin(fig_8_start_heading)],
#                 [np.sin(fig_8_start_heading), np.cos(fig_8_start_heading)]])
#     x, y = R @ np.array([x, y]) + initial_state_I[0:2]
#     vel_x, vel_y = R @ np.array([vel_x, vel_y])
#     return x, y, z, vel_x, vel_y, vel_z

class DAgger:

    def __init__(self, env, expert, policy, desired_trajectory, beta_sched, Tmax, stopping_cond, N, input_NN_size):
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
            # Begin a trajectory
            traj_steps = 0

            # period = np.random.uniform(10, 30)
            # length = period

            # xs = np.zeros((self.Tmax, 3))
            # vels = np.zeros((self.Tmax, 3))

            # t = np.arange(0, self.Tmax*self.dt, self.dt)
            # theta_des_prev = 0.0
            # x0 = np.array([0.0, 0.0, np.pi/4])
            # for i, tt in enumerate(t):
            #     x = compute_fig8_simple(period, length, tt, x0)
            #     x, y, z, vx, vy, vz = x
            #     xs[i, 0] = x
            #     xs[i, 1] = y
            #     theta_des = wrap_circular_value(np.arctan2(vy, vx))
            #     xs[i, 2] = theta_des
            #     vels[i, 0] = vx
            #     vels[i, 1] = vy
            #     vels[i, 2] = wrap_circular_value((theta_des - theta_des_prev))/self.dt
            #     theta_des_prev = theta_des

            traj_over = False
            output_reset = self.env.reset() # Reset environment to random IC at beginning

            # self.env.state[2] = np.pi/4

            x_d = self.desired_trajectory(self.Tmax, self.env.dt) # N x 2
            self.env.state = x_d[0, :].copy() # Set the initial state to the first desired state
            
            # x_I_prev = np.zeros(2)
            # theta_prev = 0.0
            obs_k = self.env.state.copy()
            while traj_steps < self.Tmax:
                # Get the expert action at the current state

                # x_I = self.env.state[0:2]
                # theta = self.env.state[2]
                # theta = wrap_circular_value(theta)
                # vel_I = (x_I - x_I_prev)/self.dt
                # omega = wrap_circular_value(theta - theta_prev)/self.dt
                                    
                # expert_action = self.expert(x_I, theta, vel_I, omega, xs[traj_steps], vels[traj_steps])

                expert_action = self.expert(obs_k, x_d[traj_steps, :])
                state_memory[sample_ind, :] = np.concatenate((obs_k, x_d[traj_steps, :]))

                # # Record the expert action and state in the memory
                # state_memory[sample_ind, :] = np.array([x_I[0],
                #                                         x_I[1],
                #                                         theta,
                #                                         vel_I[0],
                #                                         vel_I[1],
                #                                         omega,
                #                                         xs[traj_steps][0],
                #                                         xs[traj_steps][1],
                #                                         xs[traj_steps][2],
                #                                         vels[traj_steps][0],
                #                                         vels[traj_steps][1],
                #                                         vels[traj_steps][2],
                # ])
                expert_action_memory[sample_ind, :] = expert_action

                # Execute either the expert action or policy action in the environment
                if random.random() < beta:
                    obs_k, _, _, _ = self.env.step(expert_action)
                else:
                    # input_NN = np.array([x_I[0],
                    #                                     x_I[1],
                    #                                     theta,
                    #                                     vel_I[0],
                    #                                     vel_I[1],
                    #                                     omega,
                    #                                     xs[traj_steps][0],
                    #                                     xs[traj_steps][1],
                    #                                     xs[traj_steps][2],
                    #                                     vels[traj_steps][0],
                    #                                     vels[traj_steps][1],
                    #                                     vels[traj_steps][2],
                    # ])
                    input_NN = np.concatenate((obs_k, x_d[traj_steps, :]))
                    policy_action = self.policy.predict(input_NN)
                    obs_k, _, _, _ = self.env.step(policy_action.detach().numpy())

                # # Check if the trajectory reached a termination condition 
                # if result[2]:
                #     traj_over = True
                
                # x_I_prev = x_I.copy()
                # theta_prev = theta
                
                # Increment counters
                traj_steps += 1
                sample_ind += 1
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
        # Run for a specific number of iterations
        for _ in range(epochs):
            # Perform a DAgger rollout
            states, expert_actions = self.dagger_rollout()
            # Append results to the set of data to train behaviour cloning
            train_states = np.vstack((train_states, states))
            train_actions = np.vstack((train_actions, expert_actions))

            # Train the model
            self.policy.train(train_states, train_actions)
