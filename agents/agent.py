import tensorflow as tf
import tensorflow_hub as hub
from tf_agents.networks import network

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"


class AgentNetwork(network.Network):
    def __init__(self, input_tensor_spec, action_spec, name="ActorNetwork"):
        super(AgentNetwork, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        self.hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
        self.gru = tf.keras.layers.GRU(4, return_state=True)
        self.q_value_layer = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, observation, network_state=None, training=False):
        """A wrapper around `Network.call`.

        Args:
            inputs: The input to `self.call`, matching `self.input_tensor_spec`
            network_state: A state to pass to the network used by the RNN layer
            training: Optional argument to set to training mode
        Returns:
        A tuple `(outputs, new_network_state)`.
        """
        embedding = self.hub_layer(observation, training=training)
        embedding = tf.expand_dims(embedding, axis=0)
        gru_output, state = self.gru(embedding, initial_state=network_state)
        q_value = self.q_value_layer(gru_output, training=training)
        return q_value, state
