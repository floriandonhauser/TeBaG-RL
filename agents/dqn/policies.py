from typing import Any, Dict, List, Optional, Type

import tensorflow as tf
import tensorflow_hub as hub
from tf_agents.environments.tf_py_environment import tensor_spec
from tf_agents.networks import network, sequential

from agents.common.policies import BasePolicy
from agents.common.util import Schedule


class QNetwork(BasePolicy):
    """
    Q-Value network for DQN with RNN features

    :param input_spec: Input/Observation specification
    :param action_spec: Action specification
    """

    def __init__(self, observation_spec, action_spec):
        super().__init__(observation_spec, action_spec)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        fc_layer_params = (100, 50)

        # Define a helper function to create Dense layers configured with the right
        # activation and kernel initializer.
        def dense_layer(num_units):
            return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"
                ),
            )

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # it's output.
        dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2),
        )
        self.q_net = tf.keras.Sequential()
        self.q_net.add(dense_layers)
        self.q_net.add(q_values_layer)

    def call(self, observation, network_state=None, deterministic=False):
        """A wrapper around `Network.call`.

        :param observation: The input to `self.call`, matching `self.input_spec`
        :param network_state: The state of the RNN
        :param deterministic:
        :returns: A tuple `(outputs, new_network_state)`.
        """
        training = not deterministic

        q_values = self.q_net(observation)

        return q_values, network_state

    def _predict(self, observation: tf.Tensor, network_state=()):
        """
        Greedy predict policy
        :param observation: Observation
        :param network_state: Some hidden state of the RNN network
        :return: Greedy best action
        """

        q_values = self.__call__(observation, network_state, False)
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class DQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN


    :param input_spec: Input/Observation specification
    :param action_spec: Action specification
    :param num_verb: Number of verbs in the action space
    :param num_obj: Number of objects in the action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param optimizer_class: The optimizer to use,
        ``tf.optimizer.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_spec: tensor_spec,
            action_spec: tensor_spec,
            lr_schedule: Schedule,
            optimizer_class: Type[tf.optimizers.Optimizer] = tf.optimizers.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(DQNPolicy, self).__init__(
            observation_spec,
            action_spec,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.net_args = {
            "observation_spec": self.observation_spec,
            "action_spec": self.action_spec,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), learning_rate=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        return QNetwork(**self.net_args)

    def call(self, obs: tf.Tensor, state: tf.Tensor, deterministic: bool = True) -> tf.Tensor:
        return self._predict(obs, state, deterministic=deterministic)

    def _predict(self, obs: tf.Tensor, state: tf.Tensor, deterministic: bool = True) -> tf.Tensor:
        return self.q_net._predict(obs, state, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data
