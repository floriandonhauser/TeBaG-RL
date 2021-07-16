"""Module for centralized agent creation"""

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from agents import QRnnEmbNetwork


def create_agent(env, num_verb, num_obj, learning_rate):
    train_step_counter = tf.Variable(0)

    q_net, optimizer = create_policy(env, num_verb, num_obj, learning_rate)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        train_step_counter=train_step_counter,
    )
    return agent


def create_policy(env, num_verb, num_obj, learning_rate=1e-3):
    """Policy creation for agent."""

    fc_layer_params = (100, 50)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    # q_net = HubPolicy(observation_spec, action_spec, num_verb, num_obj)
    q_net = QRnnEmbNetwork(
        observation_spec,
        action_spec,
        output_fc_layer_params=fc_layer_params,
        lstm_size=[3],
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return q_net, optimizer

