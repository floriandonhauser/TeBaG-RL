"""main script for training an agent"""

from __future__ import absolute_import, division, print_function

import os, random
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from agents import create_agent
from environments import create_environments


def compute_avg_return(environment, policy, num_episodes, batch_size):
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


def collect_step(environment, policy, buffer, batch_size):
    time_step = environment.current_time_step()
    action_step = policy.action(
        time_step, policy.get_initial_state(batch_size=batch_size)
    )
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps, batch_size):
    for _ in range(steps):
        collect_step(env, policy, buffer, batch_size)


def get_rndm_game(path_dir: str):
    valid = False
    res_dir = "/content/drive/MyDrive/DeepLearningNLP/resources/"
    while not valid:
        choice = random.choice(os.listdir(res_dir + path_dir))
        if "ulx" == choice[-3:]:
            valid = True

    return os.path.join(res_dir, path_dir, choice)


def main(
    num_iterations: int = 5000,
    learning_rate: float = 1e-3,
    initial_collect_steps: int = 100,
    collect_steps_per_iteration: int = 1,
    replay_buffer_max_length: int = 100000,
    batch_size: int = 64,
    log_interval: int = 5,
    num_eval_episodes: int = 10,
    eval_interval: int = 50,
    game_gen_interval: int = 100,
    game_gen_no: int = 10,
    plot_avg_ret: bool = True,
    debug: bool = False,
    reward_dict=None,
    env_dir=None,
):
    train_env, eval_env, num_verb, num_obj = create_environments(
        debug=debug, reward_dict=reward_dict
    )
    agent = create_agent(train_env, num_verb, num_obj, learning_rate)
    agent.initialize()

    random_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec()
    )

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length,
    )

    print("Debug Start Filling RB")

    # add random training game loop here
    if env_dir is None:
        collect_data(
            train_env, random_policy, replay_buffer, initial_collect_steps, batch_size
        )
    else:
        steps = 0
        while steps <= initial_collect_steps:
            game_path = get_rndm_game(env_dir)
            train_env_tmp, _, _, _ = create_environments(
                debug=debug,
                reward_dict=reward_dict,
                env_name=game_path,
            )
            collect_data(train_env_tmp, random_policy, replay_buffer, 100, batch_size)
            steps += 100

    print("Debug End Filling RB")

    # prepare pipeline
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # agent.train = common.function(agent.train)

    agent.train_step_counter.assign(0)

    # avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = []  # [avg_return]
    iterations = []

    # create large train env list
    if env_dir is not None:
        env_list = []
        for i in range(game_gen_no):
            game_path = get_rndm_game(env_dir)
            train_env_tmp, _, _, _ = create_environments(
                debug=debug,
                reward_dict=reward_dict,
                env_name=game_path,
            )
            env_list.append(train_env_tmp)

    # learning
    for _ in range(num_iterations):

        # TODO: add random game loop here
        """
        collect_data(
            train_env, random_policy, replay_buffer, collect_steps_per_iteration, batch_size
        )
        """
        if env_dir is None:
            collect_data(
                train_env,
                agent.collect_policy,
                replay_buffer,
                collect_steps_per_iteration,
                batch_size,
            )
            collect_data(
                train_env,
                random_policy,
                replay_buffer,
                collect_steps_per_iteration,
                batch_size,
            )
        else:
            rndm_env = random.choice(env_list)
            collect_data(
                rndm_env,
                agent.collect_policy,
                replay_buffer,
                collect_steps_per_iteration,
                batch_size,
            )
            # collect_data(rndm_env, random_policy, replay_buffer, collect_steps_per_iteration, batch_size)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print(f"step = {step}: loss = {train_loss}")

        if step % eval_interval == 0:
            if env_dir is None:
                avg_return = compute_avg_return(
                    eval_env, agent.policy, num_eval_episodes, batch_size
                )
            else:
                test_env_dir = "test" + env_dir[5:]
                # test_env_dir = env_dir
                game_path = get_rndm_game(test_env_dir)
                _, eval_env_tmp, _, _ = create_environments(
                    debug=debug,
                    reward_dict=reward_dict,
                    env_name=game_path,
                )
                avg_return = compute_avg_return(
                    eval_env_tmp, agent.policy, num_eval_episodes, batch_size
                )

            print(f"step = {step}: Average Return = {avg_return}")
            iterations.append(step)
            returns.append(avg_return)

        if step % game_gen_interval == 0:
            if env_dir is not None:
                env_list = []
                for i in range(game_gen_no):
                    game_path = get_rndm_game(env_dir)
                    train_env_tmp, _, _, _ = create_environments(
                        debug=debug,
                        reward_dict=reward_dict,
                        env_name=game_path,
                    )
                    env_list.append(train_env_tmp)

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


if __name__ == "__main__":
    main()
