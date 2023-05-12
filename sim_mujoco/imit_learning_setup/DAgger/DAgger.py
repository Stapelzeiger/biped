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
                    print("policy_action: ", policy_action)
                    result = self.env.step(policy_action.detach().numpy())
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
            states, expert_actions = self.dagger_rollout()
            # Append results to the set of data to train behavoir cloning
            train_states = np.vstack((train_states, states))
            train_actions = np.vstack((train_actions, expert_actions))

            # Train the model
            self.policy.train(train_states, train_actions)

            