import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import explained_variance, get_schedule_fn


"""
Proximal Policy Optimization algorithm (PPO) (clip version)

Paper: https://arxiv.org/abs/1707.06347
Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

:param env: The environment to learn from (if registered in Gym, can be str)
:param learning_rate: The learning rate, it can be a function
    of the current progress remaining (from 1 to 0)
:param n_steps: The number of steps to run for each environment per update
    (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
    NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
    See https://github.com/pytorch/pytorch/issues/29372
:param batch_size: Minibatch size
:param n_epochs: Number of epoch when optimizing the surrogate loss
:param gamma: Discount factor
:param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
:param clip_range: Clipping parameter, it can be a function of the current progress
    remaining (from 1 to 0).
:param clip_range_vf: Clipping parameter for the value function,
    it can be a function of the current progress remaining (from 1 to 0).
    This is a parameter specific to the OpenAI implementation. If None is passed (default),
    no clipping will be done on the value function.
    IMPORTANT: this clipping depends on the reward scaling.
:param normalize_advantage: Whether to normalize or not the advantage
:param ent_coef: Entropy coefficient for the loss calculation
:param vf_coef: Value function coefficient for the loss calculation
:param max_grad_norm: The maximum value for the gradient clipping
:param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    instead of action noise exploration (default: False)
:param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    Default: -1 (only sample at the beginning of the rollout)
:param target_kl: Limit the KL divergence between updates,
    because the clipping is not enough to prevent large update
    see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
    By default, there is no limit on the kl div.
:param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
    the reported success rate, mean episode length, and mean reward over
:param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
    debug messages
:param seed: Seed for the pseudo random generators
:param device: Device (cpu, cuda, ...) on which the code should be run.
    Setting it to auto, the code will be run on the GPU if possible.
:param _init_setup_model: Whether or not to build the network at the creation of the instance
"""

class PPOImit:
    
    def __init__(self,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.env = env,
        self.learning_rate = learning_rate,
        self.n_steps = n_steps,
        self.gamma = gamma,
        self.gae_lambda = gae_lambda,
        self.ent_coef = ent_coef,
        self.vf_coef = vf_coef,
        self.max_grad_norm = max_grad_norm,
        self.use_sde = use_sde,
        self.sde_sample_freq = sde_sample_freq,
        self.stats_window_size = stats_window_size,
        self.verbose = verbose,
        self.device = device,
        self.seed = seed,
        self._init_setup_model = False,

        # Sanity check, otherwise it will lead to noisy gradient and NaN because of the advantage normalization
        if normalize_advantage:
            assert (batch_size > 1), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (not normalize_advantage), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            
            # Check that the rollout buffer size is a multiple of the mini-batch size
            assert (buffer_size % batch_size == 0), "rollout buffer size must be a multiple of the minibatch size."
            untruncated_batches = buffer_size // batch_size 

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # TODO: adaptive KL penalty scheduling the coefficient in front

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self, rollout_buffer) -> None:
        """
        Update policy using rollout buffer 'rollout_buffer'.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())



def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def get_schedule_fn(value_schedule: Union[Schedule, float]) -> Schedule:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: Constant value of schedule function
    :return: Schedule function (can return constant value)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def constant_fn(val: float) -> Schedule:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: constant value
    :return: Constant schedule function.
    """

    def func(_):
        return val

    return func