"""Main train loop class."""

from __future__ import absolute_import, division, print_function
from abc import ABC

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from agents import create_agent
from environments import create_environments
from resources import res_path

DEFAULT_HP = {
    "learning_rate": 1e-3,
    "initial_collect_steps": 5000,
    "collect_steps_per_iteration": 1,
    "replay_buffer_max_length": 100000,
    "batch_size": 128,
    "num_eval_episodes": 1,
    "game_gen_buffer": 50,
}


class TWTrainer(ABC):
    """"""

    def __init__(
        self,
        hpar: dict = DEFAULT_HP,
        reward_dict: dict = None,
        env_dir: str = None,
        debug: bool = False,
    ):
        self._hpar = hpar
        self._debug = debug
        self._reward_dict = reward_dict
        self._env_dir = env_dir

        self._agent = None
        self._rndm_pol = None
        self._train_env = None
        self._test_env = None
        self._train_env_list = []
        self._replay_buffer = None

    def _setup_training(self):
        """"""

        self._train_env, self._test_env, num_verb, num_obj = create_environments(
            debug=self._debug, reward_dict=self._reward_dict
        )
        self._train_env_list.append(self._train_env)

        self._agent = create_agent(
            self._train_env, num_verb, num_obj, self._hpar["learning_rate"]
        )
        self._agent.initialize()

        self._rndm_pol = random_tf_policy.RandomTFPolicy(
            self._train_env.time_step_spec(), self._train_env.action_spec()
        )

        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            # FIXME? Does train env has batch size? rnn not matching state fix?
            batch_size=self._train_env.batch_size,
            max_length=self._hpar["replay_buffer_max_length"],
        )

    def _fill_replay_buffer(self):
        """"""

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
                    100,
                    self._hpar["batch_size"],
                )
                steps += 100

    def _refill_env_list(self):
        """"""

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

    def train(
        self,
        num_iterations: int = 5000,
        log_interval: int = 250,
        eval_interval: int = 50,
        game_gen_interval: int = 100,
        plot_avg_ret: bool = True,
    ):
        """"""

        self._setup_training()

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
        iterations = []

        # create large train env list
        if self._env_dir is not None:
            self._refill_env_list()

        # learning
        for _ in range(num_iterations):

            rndm_env = random.choice(self._train_env_list)
            self._collect_data(
                rndm_env,
                self._agent.collect_policy,
                self._replay_buffer,
                self._hpar["collect_steps_per_iteration"],
                self._hpar["batch_size"],
            )

            experience, unused_info = next(iterator)
            train_loss = self._agent.train(experience).loss

            step = self._agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print(f"step = {step}: loss = {train_loss}")

            if step % eval_interval == 0:
                if self._env_dir is None:
                    avg_return = self._compute_avg_return(
                        self._test_env,
                        self._agent.policy,
                        self._hpar["num_eval_episodes"],
                        self._hpar["batch_size"],
                    )
                else:
                    test_env_dir = "test" + self._env_dir[5:]

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

                print(f"step = {step}: Average Return = {avg_return}")
                iterations.append(step)
                returns.append(avg_return)

            if step % game_gen_interval == 0:
                if self._env_dir is not None:
                    self._refill_env_list()

        iterations = np.array(iterations)
        returns = np.array(returns)

        if plot_avg_ret:
            print(iterations.shape, iterations)
            print(returns.shape, returns)
            plt.plot(iterations, returns)
            plt.ylabel("Average Return")
            plt.xlabel("Iterations")
            plt.ylim(top=250)
            plt.savefig("training_curve.png")

        return returns

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

    @staticmethod
    def _collect_step(environment, policy, buffer, batch_size):
        time_step = environment.current_time_step()
        action_step = policy.action(
            time_step, policy.get_initial_state(batch_size=batch_size)
        )
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        buffer.add_batch(traj)

    def _collect_data(self, env, policy, buffer, steps, batch_size):
        for _ in range(steps):
            self._collect_step(env, policy, buffer, batch_size)


def main():
    trainer = TWTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
