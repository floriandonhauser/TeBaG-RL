import tensorflow as tf
import tensorflow_hub as hub
from tf_agents.networks import network

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"


class HubPolicy(network.Network):
    def __init__(self, input_tensor_spec, action_spec, num_verb, num_obj, name="ActorNetwork"):
        super().__init__()

        num_actions = action_spec.maximum - action_spec.minimum + 1
        assert num_actions == num_verb * num_obj
        self.num_verb = num_verb
        self.num_obj = num_obj

        self.hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
        self.gru = tf.keras.layers.GRU(4, return_state=True)

        self.verb_layer = tf.keras.layers.Dense(num_verb, activation=None)
        self.obj_layer = tf.keras.layers.Dense(num_obj, activation=None)

        self.number_of_strings = input_tensor_spec.shape[0]

    def call(self, observation, network_state=(), training=False):
        """A wrapper around `Network.call`.

        Args:
            inputs: The input to `self.call`, matching `self.input_tensor_spec`
            network_state: A state to pass to the network used by the RNN layer
            training: Optional argument to set to training mode
        Returns:
        A tuple `(outputs, new_network_state)`.
        """
        if network_state is not None and len(network_state) == 0:
            network_state = None

        flattened_observation = tf.reshape(observation, (-1))
        embedded_observations = self.hub_layer(flattened_observation, training=training)
        embedded_observations = tf.reshape(embedded_observations, (observation.shape[0], observation.shape[1], 50))
        gru_output, network_state = self.gru(embedded_observations, initial_state=network_state)
        gru_output = tf.expand_dims(gru_output, axis=1)
        verb_q_value = self.verb_layer(gru_output, training=training)
        obj_q_value = self.obj_layer(gru_output, training=training)
        q_value_multiplied = tf.matmul(verb_q_value, obj_q_value, transpose_a=True)
        q_values = tf.reshape(q_value_multiplied, (observation.shape[0], -1))

        return q_values, ()
