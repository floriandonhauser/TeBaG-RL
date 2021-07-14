"""Policies: abstract base class and concrete implementations."""

import collections
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment


class BaseModel(tf.Module, ABC):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    :param observation_spec: The observation space of the environment
    :param action_spec: The action space of the environment
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_spec: TFPyEnvironment,
            action_spec: TFPyEnvironment,
            optimizer_class: Type[tf.optimizers.Optimizer] = tf.optimizers.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(BaseModel, self).__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.observation_spec = observation_spec
        self.action_spec = action_spec

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[tf.optimizers.Optimizer]

    @abstractmethod
    def call(self, *args, **kwargs):
        pass


class BasePolicy(BaseModel):
    """The base policy object.

    Parameters are mostly the same as `BaseModel`; additions are documented below.

    :param args: positional arguments passed through to `BaseModel`.
    :param kwargs: keyword arguments passed through to `BaseModel`.
    """

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super(BasePolicy, self).__init__(*args, **kwargs)

    @abstractmethod
    def _predict(self, observation: tf.Tensor, state: tf.Tensor, deterministic: bool = False) -> tf.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param state:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if state is None:
            state = ()

        actions, state = self._predict(observation, state, deterministic=deterministic)

        return actions, state
