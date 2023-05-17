import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

class Agent(nn.Module):

    def __init__(self, envs, actor_arch, critic_arch):
        super(Agent, self).__init__()
        state_size = np.array(envs.single_observation_space.shape).prod()
        act_size = np.array(envs.single_action_space.shape).prod()

        assert actor_arch[0]['Input'] == state_size, "Actor network input does not match size of states"
        assert actor_arch[-1]['Output'] == act_size, "Actor network output does not match size of actions"
        assert critic_arch[0]['Input'] == state_size, "Critic network input does not match size of states"
        assert critic_arch[-1]['Output'] == 1, "Critic network output is not scalar"

        self.actor_mean = Agent._gen_policy(actor_arch)
        self.actor_logstd = nn.Parameter(torch.zeros(act_size))
        self.critic = Agent._gen_policy(critic_arch)

        self.eval_mode = False

    staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    staticmethod
    def _gen_policy(policy_architecture):
        """Generates a torch.nn Sequential model from the policy architecture

        Args:
            policy_architecture (iterable): Iterable definiting policy architecture
                with 'Layer', 'Input', 'Output', and other arguments for each layer
        """
        layers = []
        for layer in policy_architecture:
            if layer['Layer'] == 'Linear':
                if 'std' in layer.keys():
                    # Use a specific standard deviation for orthogonal initialization
                    layers.append(Agent.layer_init(nn.Linear(layer['Input'], layer['Output']), std=layer['std']))
                else:
                    # Otherwise use the default
                    layers.append(Agent.layer_init(nn.Linear(layer['Input'], layer['Output'])))
            elif layer['Layer'] == 'ReLU':
                layers.append(nn.ReLU())
            elif layer['Layer'] == 'Tanh':
                layers.append(nn.Tanh())
            else:
                ValueError(f"Layer type not recognized: {layer}")
        model = nn.Sequential(*layers)
        return model
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        if self.eval_mode:
            return action_mean
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def save_policy(self, path):
        """Saves the current policy weights to a designated path

        Args:
            path (path-like): location to save the current policy weights
        """
        torch.save(self.state_dict(), path)

    def load_policy(self, path):
        """Loads policy weights from the designated path

        Args:
            path (path-like): location to read policy weights
        """
        self.load_state_dict(torch.load(path))

    def set_eval_mode(self, eval_mode):
        self.eval_mode = eval_mode


class PPO:

    def __init__(
            self, envs, actor_arch, critic_arch, lr=0.0003, epochs=10, num_rollouts=2,
            num_rollout_steps=2048, gamma=0.99, gae_lambda=0.95, num_mb=32, clip_coeff=0.2,
            ent_coeff=0.0001, vf_coeff=0.5, max_grad_norm=0.5, norm_adv=True, clip_vloss=True,
            anneal_lr=True, anneal_ent=True, target_kl=None, rng_seed=None, run_name=None
    ):
        # Get device for training
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # Initialize the agent, put on training device
        self.agent = Agent(envs, actor_arch, critic_arch).to(self.device)
        self.state_size = envs.single_observation_space.shape
        self.act_size = envs.single_action_space.shape

        # Initialize a bunch of variables
        self.envs = envs
        self.num_roll = num_rollouts
        self.roll_steps = num_rollout_steps
        self.batch_size = self.num_roll * self.roll_steps
        self.minibatch_size = int(self.batch_size // num_mb)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coeff = clip_coeff
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.anneal_lr = anneal_lr
        self.anneal_ent = anneal_ent

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        self.lr = lr
        self.epochs = epochs

        self.total_steps = 0
        self.total_rollouts = 0

        # Tensorboard stuff
        if run_name is not None:
            self.writer = SummaryWriter(f"runs/{run_name}")
        else:
            self.writer = SummaryWriter()

        # Random seeding for reproducability
        if rng_seed is not None:
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)
            torch.backends.cudnn.deterministic = True
    
    def train(self, iterations):
        self.start_time = time.time()
        ent_coeff = self.ent_coeff
        for ii in range(iterations):
            frac = 1.0 - ii / iterations
            if self.anneal_lr:
                lrnow = frac * self.lr
                self.optimizer.param_groups[0]["lr"] = lrnow
            if self.anneal_ent:
                self.ent_coeff = ent_coeff * frac
            states, actions, logprobs, advantages, returns, values = self.collect_rollouts()

            self.learn(states, actions, logprobs, advantages, returns, values)
        
        self.envs.close()
        self.writer.close()

    def collect_rollouts(self):
        states = torch.zeros((self.roll_steps, self.num_roll) + self.state_size).to(self.device)
        actions = torch.zeros((self.roll_steps, self.num_roll) + self.act_size).to(self.device)
        logprobs = torch.zeros((self.roll_steps, self.num_roll)).to(self.device)
        rewards = torch.zeros((self.roll_steps, self.num_roll)).to(self.device)
        dones = torch.zeros((self.roll_steps, self.num_roll)).to(self.device)
        values = torch.zeros((self.roll_steps, self.num_roll)).to(self.device)

        steps = 0

        next_state = torch.Tensor(self.envs.reset()).to(self.device)
        next_done = torch.zeros(1)

        # Roll out the policy
        for step in range(self.roll_steps):
            self.total_steps += self.num_roll
            # Store the next state and done flag
            states[step] = next_state
            dones[step] = next_done

            # Query action and value, no gradient computations required
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_state)
                values[steps] = value.flatten()

            # Store actions, logprobs, rewards
            actions[step] = action
            logprobs[step] = logprob

            next_state, reward, done, info = self.envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(self.device).view(-1)

            # Format next state and done flag as a tensor
            next_state, next_done = torch.Tensor(next_state).to(self.device), torch.Tensor(done).to(self.device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={self.total_steps}, episodic_return={item['episode']['r']}")
                    self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.total_steps)
                    self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.total_steps)
                    break

        # Compute GAE
        with torch.no_grad():
            next_value = self.agent.get_value(next_state).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.roll_steps)):
                if t == self.roll_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # flatten the batch
        b_states = states.reshape((-1,) + self.state_size)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.act_size)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        return b_states, b_actions, b_logprobs, b_advantages, b_returns, b_values

    def learn(self, states, actions, logprobs, advantages, returns, values):
        inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(states[mb_inds], actions[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coeff).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -self.clip_coeff,
                        self.clip_coeff,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                # Total Loss
                loss = pg_loss - self.ent_coeff * entropy_loss + v_loss * self.vf_coeff

                # Perform backpropogation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)

                self.optimizer.step()
            
            # Early stopping based on KL divergence
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break
        
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record metrics for plotting/analysis
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.total_steps)
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.total_steps)
        self.writer.add_scalar("charts/ent_weight", self.ent_coeff, self.total_steps)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.total_steps)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.total_steps)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.total_steps)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.total_steps)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.total_steps)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.total_steps)
        print("SPS:", int(self.total_steps / (time.time() - self.start_time)))
        self.writer.add_scalar("charts/SPS", int(self.total_steps / (time.time() - self.start_time)), self.total_steps)
    
    def save_policy(self, path):
        """Saves the current policy weights to a designated path

        Args:
            path (path-like): location to save the current policy weights
        """
        torch.save(self.agent.state_dict(), path)

    def load_policy(self, path):
        """Loads policy weights from the designated path

        Args:
            path (path-like): location to read policy weights
        """
        self.agent.load_state_dict(torch.load(path))







