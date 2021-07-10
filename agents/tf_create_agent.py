"""Module for centralized agent creation"""

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

from agents import HubPolicy


def create_agent(env, num_verb, num_obj, learning_rate):
    train_step_counter = tf.Variable(0)

    q_net, optimizer = create_policy(env, num_verb, num_obj, learning_rate)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        # TODO which loss function? default is element_wise_huber_loss
        # td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
    )
    return agent


def create_policy(env, num_verb, num_obj, learning_rate=1e-3):
    """Policy creation for agent."""

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    q_net = HubPolicy(observation_spec, action_spec, num_verb, num_obj)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return q_net, optimizer
