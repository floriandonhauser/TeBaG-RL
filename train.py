"""main script for training an agent"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from agents import HubPolicy
from environments import TWGameEnv

import numpy as np

# Configuration
num_iterations = 5

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
log_interval = 1

num_eval_episodes = 2
eval_interval = 1


# Environments (Max)
def create_environments():
    env_name = "resources/rewardsDense_goalBrief.ulx"
    path_verbs = "resources/words_verbs_short.txt"
    path_objs = "resources/words_objs_short.txt"
    path_badact = "resources/bad_actions.txt"
    train_py_env = TWGameEnv(
        game_path=env_name,
        path_verb=path_verbs,
        path_obj=path_objs,
        path_badact=path_badact,
        debug=True,
        flatten_actspec=True,
    )
    eval_py_env = TWGameEnv(
        game_path=env_name,
        path_verb=path_verbs,
        path_obj=path_objs,
        path_badact=path_badact,
        debug=False,
        flatten_actspec=True,
    )

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    return train_env, eval_env, train_py_env.num_verb, train_py_env.num_obj


# policy (Florian)
def create_policy(env, num_verb, num_obj, learning_rate=1e-3):
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    q_net = HubPolicy(observation_spec, action_spec, num_verb, num_obj)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return q_net, optimizer


def create_agent(env, num_verb, num_obj):
    train_step_counter = tf.Variable(0)

    q_net, optimizer = create_policy(env, num_verb, num_obj)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,  # TODO which loss function?
        train_step_counter=train_step_counter)
    return agent


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


def main():
    train_env, eval_env, num_verb, num_obj = create_environments()
    agent = create_agent(train_env, num_verb, num_obj)
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

    #avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [] # [avg_return]

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
            print(f'step = {step}: Average Return = {avg_return}')
            returns.append(avg_return)

    iterations = np.arange(1, num_iterations+1, eval_interval) # range(0, num_iterations + 1, eval_interval)
    returns = np.array(returns)
    print(iterations.shape, iterations)
    print(returns.shape, returns)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.savefig('training_curve.png')


if __name__ == "__main__":
    main()
