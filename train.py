"""main script for training an agent"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from agents import create_agent
from environments import create_environments

import numpy as np


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


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
    plot_avg_ret: bool = True,
    debug: bool = False,
):
    train_env, eval_env, num_verb, num_obj = create_environments(debug=debug)
    agent = create_agent(train_env, num_verb, num_obj, learning_rate)
    agent.initialize()

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    # prepare pipeline
    dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                       sample_batch_size=batch_size,
                                       num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # some optimization
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    # avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = []  # [avg_return]
    iterations = []

    # learning
    for _ in range(num_iterations):

        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print(f'step = {step}: loss = {train_loss}')

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print(f"step = {step}: Average Return = {avg_return}")
            iterations.append(step)
            returns.append(avg_return)

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
