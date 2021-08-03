"""Module for centralized agent creation"""

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from agents import HubPolicyFC, HubPolicyBert


def create_agent(env, num_verb, num_obj, learning_rate, agent_tag):
    train_step_counter = tf.Variable(0, dtype=tf.int64)

    q_net, optimizer = create_policy(env, num_verb, num_obj, learning_rate, agent_tag)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        train_step_counter=train_step_counter,
    )
    return agent


def create_policy(env, num_verb, num_obj, learning_rate=1e-3, agent_tag=None):
    """Policy creation for agent."""

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    if agent_tag is None:
        agent_tag = "FCPolicy"
    if agent_tag == "FCPolicy":
        q_net = HubPolicyFC(observation_spec, action_spec, num_verb, num_obj)
    elif agent_tag == "BertPolicy":
        q_net = HubPolicyBert(observation_spec, action_spec, num_verb, num_obj)
    else:
        ValueError
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return q_net, optimizer
