"""Main train loop class."""

from __future__ import absolute_import, division, print_function
from abc import ABC

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from agents import create_agent
from environments import create_environments
from resources import res_path, DEFAULT_PATHS

DEFAULT_HP = {
    "learning_rate": 1e-3,
    "initial_collect_steps": 5000,
    "collect_steps_per_iteration": 1,
    "replay_buffer_max_length": 100000,
    "batch_size": 128,
    "num_eval_episodes": 1,
    "num_eval_games": 10,
    "num_test_games": 50,
    "game_gen_buffer": 50,
}


class TWTrainer(ABC):
    """Trainer for TextWorld RL agent

    Parameters:
    -----------
    hpar: dict
        Dictionary of hyper parameters.
    reward_dict: dict
        Dictionary of reward values to be used for reward calculation
    env_dir: str
        Path to game directory to pull random game files from. If set to None, debug
        game is used instead.
    debug: bool
        Enables debug mode with testing and outputs
    biased_buffer: bool
        Enables biased replay buffer. For given threshold and probabilities, only
        desired trajectories are added to the replay buffer and other cases with a
        certain probability. Agent will only be trained every 10th iteration to avoid
        overfitting on too small samples, however, num_iterations will be adjusted
        automatically in train() method.
    agent_label: str
        Tag to chose different q networks for the policy.
        Implemented options ["BertPolicy", "FCPolicy"].
    """

    def __init__(
        self,
        hpar: dict = DEFAULT_HP,
        reward_dict: dict = None,
        env_dir: str = None,
        debug: bool = False,
        biased_buffer=False,
        agent_label: str = None,
    ):
        self._hpar = hpar
        self._agent_label = agent_label
        self._debug = debug
        self._reward_dict = reward_dict
        self._env_dir = env_dir
        self._biased_buffer = biased_buffer
        # Reward values to be distinguished by biased buffer
        self._biased_buffer_thr = tf.constant([10.0, 0.0, -30.0], dtype=np.float32)
        # Probabilities to be not counted for the last two cases above for biased buffer
        self._biased_buffer_accept_prob = tf.constant([0.66, 0.99], dtype=np.float32)

        self._agent = None
        self._rndm_pol = None
        self._train_env = None
        self._test_env = None
        self._train_env_list = []
        self._replay_buffer = None
        self.summary_writer = tf.summary.create_file_writer(
            DEFAULT_PATHS["path_logdir"]
        )
        self.summary_writer.set_as_default()

        self._setup_training()

    def _setup_training(self):
        """Instantiating all relevant entities for training"""

        self._train_env, self._test_env, num_verb, num_obj = create_environments(
            debug=self._debug, reward_dict=self._reward_dict
        )
        self._train_env_list.append(self._train_env)

        self._agent = create_agent(
            self._train_env,
            num_verb,
            num_obj,
            self._hpar["learning_rate"],
            self._agent_label,
        )
        self._agent.initialize()

        self._rndm_pol = random_tf_policy.RandomTFPolicy(
            self._train_env.time_step_spec(), self._train_env.action_spec()
        )

        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            # FIXME? Does train env has batch size?
            batch_size=self._train_env.batch_size,
            max_length=self._hpar["replay_buffer_max_length"],
        )

    def _fill_replay_buffer(self, use_env_buffer: bool = True):
        """Fill replay buffer with random agent."""

        if self._env_dir is None:
            self._collect_data(
                self._train_env,
                self._rndm_pol,
                self._replay_buffer,
                self._hpar["initial_collect_steps"],
                self._hpar["batch_size"],
            )
        else:
            steps = 0
            while steps <= self._hpar["initial_collect_steps"]:
                if use_env_buffer:
                    train_env_tmp = random.choice(self._train_env_list)
                else:
                    game_path = self._get_rndm_game(self._env_dir)
                    train_env_tmp, _, _, _ = create_environments(
                        debug=self._debug,
                        reward_dict=self._reward_dict,
                        env_name=game_path,
                        onlytrain=True,
                    )
                self._collect_data(
                    train_env_tmp,
                    self._rndm_pol,
                    self._replay_buffer,
                    500,
                    self._hpar["batch_size"],
                )
                steps += 500

    def _refill_env_list(self):
        """Fill _train_env_list with a number of generated environments to train on."""

        self._train_env_list = []
        for i in range(self._hpar["game_gen_buffer"]):
            game_path = self._get_rndm_game(self._env_dir)
            train_env_tmp, _, _, _ = create_environments(
                debug=self._debug,
                reward_dict=self._reward_dict,
                env_name=game_path,
                onlytrain=True,
            )
            self._train_env_list.append(train_env_tmp)

    def change_env_dir(self, dir_name: str):
        """Change directory to get (random) game files from.

        Parameters:
        -----------
        dir_name: str
            Name of game directory. If set to None, defaulting to single debug game.
        """

        self._env_dir = dir_name

    def train(
        self,
        num_iterations: int = 5000,
        train_interval: int = 10,
        log_interval: int = 250,
        eval_interval: int = 50,
        game_gen_interval: int = 100,
        continue_training=False,
        rndm_fill_replay=True,
        plot_avg_ret: bool = True,
        test_agent: bool =False,
    ):
        """Central training loop.

        Parameters:
        -----------
        num_iterations: int
            Number of training steps to train the agent.
        train_interval: int
            Interval to train the agent in case of biased replay buffer.
        log_interval: int
            Interval to print current loss and replay buffer size.
        eval_interval: int
            Interval to calculate average reward of current agent.
        game_gen_interval: int
            Interval to refill list of created game environments to train agent with.
        continue_training: bool
            Continues training instead of resetting replay buffer and agent.
        rndm_fill_replay: bool
            Toggle filling replay agent with random agent before training.
        plot_avg_ret: bool
            Plot eval scores at the end of training loop.
        test_agent: bool
            Evaluate agent at the end of training on random sample from the test set.
        """

        if not continue_training:
            self._setup_training()

        # create large train env list
        if self._env_dir is not None:
            self._refill_env_list()

        if rndm_fill_replay:
            self._fill_replay_buffer()

        # prepare pipeline
        dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self._hpar["batch_size"],
            num_steps=2,
        ).prefetch(3)

        iterator = iter(dataset)

        self._agent.train_step_counter.assign(0)

        returns = []
        returns_buffer = []
        iterations = []


        # learning
        if self._biased_buffer:
            num_iterations = train_interval * num_iterations

        for trials in range(num_iterations):

            rndm_env = random.choice(self._train_env_list)
            self._collect_data(
                rndm_env,
                self._agent.collect_policy,
                self._replay_buffer,
                self._hpar["collect_steps_per_iteration"],
                self._hpar["batch_size"],
            )

            # Update agent occasionally, as data generation too slow with biased buffer
            if self._biased_buffer:
                if not trials % train_interval == 0:
                    continue

            experience, unused_info = next(iterator)
            train_loss = self._agent.train(experience).loss

            step = self._agent.train_step_counter.numpy()

            tf.summary.scalar("loss", train_loss, step=step)
            tf.summary.scalar("BufferSize", self._replay_buffer.num_frames(), step=step)

            if step % log_interval == 0:
                print(
                    f"step = {step}: loss = {train_loss:0.2e}, Buff-len = {self._replay_buffer.num_frames()}"
                )

            if step % eval_interval == 0:
                if self._env_dir is None:
                    avg_return = self._compute_avg_return(
                        self._test_env,
                        self._agent.policy,
                        self._hpar["num_eval_episodes"],
                        self._hpar["batch_size"],
                    )
                    avg_return_currbuff = avg_return
                else:
                    # test_env_dir = "test" + self._env_dir[5:]
                    # Use train dir!
                    test_env_dir = self._env_dir
                    eval_res_all = 0.0
                    eval_res_currbuff = 0.0
                    for n_eval in range(self._hpar["num_eval_games"]):
                        game_path = self._get_rndm_game(test_env_dir)
                        _, eval_env_tmp, _, _ = create_environments(
                            debug=self._debug,
                            reward_dict=self._reward_dict,
                            env_name=game_path,
                        )
                        avg_return = self._compute_avg_return(
                            eval_env_tmp,
                            self._agent.policy,
                            self._hpar["num_eval_episodes"],
                            self._hpar["batch_size"],
                        )
                        eval_res_all += avg_return

                        """
                        rndm_env = random.choice(self._train_env_list)
                        avg_return = self._compute_avg_return(
                            rndm_env,
                            self._agent.policy,
                            self._hpar["num_eval_episodes"],
                            self._hpar["batch_size"],
                        )
                        eval_res_currbuff += avg_return
                        """
                    for env in self._train_env_list:
                        avg_return = self._compute_avg_return(
                            env,
                            self._agent.policy,
                            self._hpar["num_eval_episodes"],
                            self._hpar["batch_size"],
                        )
                        eval_res_currbuff += avg_return

                    avg_return = eval_res_all / self._hpar["num_eval_games"]
                    avg_return_currbuff = eval_res_currbuff / len(self._train_env_list)
                    # avg_return_currbuff = eval_res_currbuff / self._hpar["num_eval_games"]

                tf.summary.scalar("eval_score_all", avg_return, step=step)
                tf.summary.scalar("eval_score_currbuff", avg_return_currbuff, step=step)
                print(f"step = {step}: Average Return (all games / curr buff) = {avg_return} / {avg_return_currbuff}")
                iterations.append(step)
                returns.append(avg_return)
                returns_buffer.append(avg_return_currbuff)

            if step % game_gen_interval == 0:
                if self._env_dir is not None:
                    self._refill_env_list()

        if test_agent:
            test_env_dir = "test" + self._env_dir[5:]
            test_res = 0.0
            rndm_res = 0.0
            for n_eval in range(self._hpar["num_test_games"]):
                game_path = self._get_rndm_game(test_env_dir)
                _, eval_env_tmp, _, _ = create_environments(
                    debug=self._debug,
                    reward_dict=self._reward_dict,
                    env_name=game_path,
                )
                avg_return_agent = self._compute_avg_return(
                    eval_env_tmp,
                    self._agent.policy,
                    self._hpar["num_eval_episodes"],
                    self._hpar["batch_size"],
                )
                avg_return_rndm = self._compute_avg_return(
                    eval_env_tmp,
                    self._rndm_pol,
                    self._hpar["num_eval_episodes"],
                    self._hpar["batch_size"],
                )
                test_res += avg_return_agent
                rndm_res += avg_return_rndm

            test_res = test_res / self._hpar["num_test_games"]
            rndm_res = rndm_res / self._hpar["num_test_games"]
            print(f"Agent performance: {test_res} \t vs. random agent: {rndm_res}")

        iterations = np.array(iterations)
        returns = np.array(returns)
        returns_buffer = np.array(returns_buffer)

        if plot_avg_ret:
            fig, axs = plt.subplots(1, 1)
            fig.suptitle("Training Rewards")
            # fig.subplots_adjust(hspace=0.6)
            axs.plot(iterations, returns, label="All games")
            axs.set(xlabel="Iterations [ ]", ylabel="AvgReward [ ]")
            axs.plot(iterations, returns_buffer, label="CurrBuff games")
            if test_agent:
                axs.plot(iterations, [test_res]*len(iterations), label="Test games")
            axs.set_ylim([-120, 220])
            axs.legend()

            # plt.tight_layout()
            plt.savefig("training_curve.png")

        return returns, returns_buffer

    @staticmethod
    def _compute_avg_return(environment, policy, num_episodes, batch_size):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(
                    time_step, policy.get_initial_state(batch_size=batch_size)
                )
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    @staticmethod
    def _get_rndm_game(path_dir: str):
        valid = False
        while not valid:
            choice = random.choice(os.listdir(res_path + "/" + path_dir))
            if "ulx" == choice[-3:]:
                valid = True

        return os.path.join(res_path, path_dir, choice)

    def _collect_step(self, environment, policy, buffer, batch_size):
        time_step = environment.current_time_step()
        action_step = policy.action(
            time_step, policy.get_initial_state(batch_size=batch_size)
        )
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        if self._biased_buffer:
            # Accept high reward immediately
            if tf.math.greater(next_time_step.reward, self._biased_buffer_thr[0]):
                buffer.add_batch(traj)
            # Accept high negative reward immediately
            elif tf.math.greater(self._biased_buffer_thr[2], next_time_step.reward):
                buffer.add_batch(traj)
            else:
                # Throw dice, if accept bad result by chance.
                dice_res = tf.random.uniform(
                    (1,), minval=0.0, maxval=1.0, dtype=tf.float32
                )
                # Accept reward > 0 with high probability
                if tf.math.greater(next_time_step.reward, self._biased_buffer_thr[1]):
                    if dice_res > self._biased_buffer_accept_prob[0]:
                        buffer.add_batch(traj)
                # Accept reward < 0 with low probability
                else:
                    if dice_res > self._biased_buffer_accept_prob[1]:
                        buffer.add_batch(traj)

        else:
            buffer.add_batch(traj)

    def _collect_data(self, env, policy, buffer, steps, batch_size):
        for _ in range(steps):
            self._collect_step(env, policy, buffer, batch_size)


def main():
    HP = {
        "learning_rate": 1e-3,
        "initial_collect_steps": 2000,
        "collect_steps_per_iteration": 1,
        "replay_buffer_max_length": 100000,
        # large values lead to OOM with bert policy
        "batch_size": 64,
        "num_eval_episodes": 1,
        "game_gen_buffer": 25,
        "num_eval_games": 10,
    }
    REWARDS = {
        "win_lose_value": 100,
        "max_loop_pun": 0,
        "change_reward": 1,
        "useless_act_pun": 1,
        "cmd_in_adm": 1,
    }

    trainer = TWTrainer(
        env_dir="train_games_lvl2",
        reward_dict=REWARDS,
        hpar=HP,
        debug=False,
        biased_buffer=True,
        # embedding into fc is default policy
        # agent_label="FCPolicy",
        agent_label="BertPolicy",
    )
    trainer.train(
        num_iterations=5000,
        log_interval=250,
        eval_interval=250,
        game_gen_interval=500,
        plot_avg_ret=True,
    )


if __name__ == "__main__":
    main()
