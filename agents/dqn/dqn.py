import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec

from agents.common.util import Schedule, get_linear_fn, get_schedule_fn, set_random_seed, update_learning_rate
from agents.dqn.policies import DQNPolicy


class DQN:
    """
    Deep Q Learning inspired by my code on stable-baselines3

    :param policy: The policy model to use
    :param env: The environment to learn from
    :param learning_rate: Can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions from before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)

    """

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: TFPyEnvironment,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):

        self.policy_class = policy
        self.env = env
        self.eval_env = None
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_spec = None
        self.action_spec = None
        self.n_envs = 1
        self.num_timesteps = tf.Variable(0)

        self.seed = seed
        self.learning_rate = learning_rate
        self.lr_schedule = None
        self._last_obs = None
        self._last_dones = None
        self._episode_num = tf.Variable(0)
        # Track the training progress from [1,0] used for updating the learning rate
        self._current_progress_remaining = 1

        # from off_policy_algorithm

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps

        # Save train freq parameter
        self.train_freq = train_freq
        self.replay_buffer = None  # type: Optional[ReplayBuffer]

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        if _init_setup_model:
            self._setup_model()

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_learning_rate(self, optimizers: Union[List[tf.optimizers.Optimizer], tf.optimizers.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: TFPyEnvironment,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> int:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()

        if reset_num_timesteps:
            self.num_timesteps.assign(0)
            self._episode_num.assign(0)
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # TODO reset like this valid for TFEnvironment?
            self._last_dones = np.zeros((self.env.num_envs,), dtype=bool)

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = eval_env

        # Configure logger's outputs
        # utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        return total_timesteps

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.eval_env is not None:
            self.eval_env.seed(seed)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.collect_data_spec, batch_size=self.batch_size, max_lenght=self.buffer_size
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_spec,
            self.action_spec,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )

        # Convert train freq parameter to TrainFreq object
        # self._convert_train_freq() TODO
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        dataset = self.replay_buffer.as_dataset()

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer TODO different replaybuffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Compute the next Q-values using the target network
            next_q_values = self.q_net_target(replay_data.next_observations)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            # 1-step TD target
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            with tf.GradientTape() as tape:
                current_q_values = self.q_net(replay_data.observations)

                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = tf.gather(current_q_values, dim=1, index=replay_data.actions.long())

                loss = tf.keras.losses.huber(target_q_values, current_q_values)

            train_params = self.q_net.trainable_weights

            grad = tape.gradient(loss, train_params)
            # TODO clip gradients

            self.policy.optimizer.apply_gradients(zip(grad, train_params))

            losses.append(loss.item())

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            rand_policy = random_py_policy.RandomPyPolicy(time_step_spec=None, action_spec=self.action_spec)
            action = rand_policy.action(None)  # TODO maybe speedup?
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 4,
        eval_env: Optional[TFPyEnvironment] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):

        total_timesteps = self._setup_learn(
            total_timesteps, eval_env, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
        return self
