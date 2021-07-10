from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from agents.dqn.dqn import DQN
from agents.dqn.policies import DQNPolicy

num_iterations = 20000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
env.reset()
print(type(env))

train_py_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
eval_py_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

agent = DQN(DQNPolicy, train_py_env, 0.4, 10000, 1000, target_update_interval=100)

agent.learn(1000, eval_env=eval_py_env)
