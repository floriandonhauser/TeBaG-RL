from typing import Tuple

import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.environments import random_py_environment, tf_py_environment
from tf_agents.specs import array_spec

from agents import HubPolicy


def create_policy(env) -> Tuple[tf_agents.networks.Network, tf_agents.typing.types.Optimizer]:
    learning_rate = 1e-3
    q_net = HubPolicy(env.observation_spec(), env.action_spec())
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return q_net, optimizer


def test_agent():
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
        print("Changed memory growth for", physical_devices[i])

    print("Running main in hub_policy.py")
    action_spec = array_spec.BoundedArraySpec((1,), np.int, minimum=0, maximum=10)
    observation_spec = array_spec.ArraySpec((1,), np.str)
    random_env = random_py_environment.RandomPyEnvironment(observation_spec=observation_spec, action_spec=action_spec)
    # Convert the environment to a TFEnv to generate tensors.
    tf_env = tf_py_environment.TFPyEnvironment(random_env)
    print("Created environment")
    # Create the policy as a q_net and an optimizer
    q_net, optimizer = create_policy(tf_env)
    print("Created q_net")
    # Test the q_net with a test input
    # TODO Maybe the random environment can create strings?
    # Right now it fails if I simply calltf_env.reset()
    observation = tf.constant(["This is a test string"])
    q_value, state = q_net(observation)
    print("the output")
    print(q_value)
    print("done")
