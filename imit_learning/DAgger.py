import numpy as np
import random


class DAgger:

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
        self.policy = policy                # policy network
        self.dt = self.env.dt               # Time step of the environment

    def dagger_rollout(self) -> tuple:
        print(f"Collecting Rollouts {self.rollout_iters}")
        # Initialize memory for states and expert actions
        state_memory = np.zeros((self.Tmax * self.N, 2*self.env.xdim))
        expert_action_memory = np.zeros((self.Tmax * self.N, self.env.udim))
        # Grab the beta weighting expert vs. policy
        beta = self.beta_schedule[self.rollout_iters]

        # Run a Dagger Rollout
        sample_ind = 0
        for _ in range(self.N):
            # Begin a trajectory
            traj_steps = 0

            frequency = np.random.uniform(0.05, 0.10,)
            def get_N_pt_traj(N, frequency, dt):
                des_traj_pos = lambda t: np.cos(t*frequency)
                pos_des = [des_traj_pos(i) for i in range(N)]
                des_traj_vel = np.zeros(N)

                for i in range(N - 1):
                    des_traj_vel[i] = (pos_des[i+1] - pos_des[i])/dt
                return pos_des, des_traj_vel

            des_traj_pos, des_traj_vel = get_N_pt_traj(self.Tmax, frequency, self.dt)

            traj_over = False
            output_reset = self.env.reset() # Reset environment to random IC at beginning
            self.env.state = np.array([des_traj_pos[0], des_traj_vel[0]])
            obs_k = self.env.state.copy()
            while traj_steps < self.Tmax:
                # Get the expert action at the current state
                des_reg_pt = np.array([des_traj_pos[traj_steps], des_traj_vel[traj_steps]])
                expert_action = self.expert(obs_k, des_reg_pt)

                # Record the expert action and state in the memory
                state_memory[sample_ind, :] = np.concatenate((obs_k, des_reg_pt))
                expert_action_memory[sample_ind, :] = expert_action

                # Execute either the expert action or policy action in the environment
                if random.random() < beta:
                    obs_k, _, _, _ = self.env.step(expert_action)
                else:
                    input_NN = np.concatenate((obs_k, des_reg_pt))
                    policy_action = self.policy.predict(input_NN)
                    obs_k, _, _, _ = self.env.step(policy_action.detach().numpy()[0])

                # # Check if the trajectory reached a termination condition 
                # if result[2]:
                #     traj_over = True
                
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
        train_states = np.zeros((0, 2*self.env.xdim))
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
