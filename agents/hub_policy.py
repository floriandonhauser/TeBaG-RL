import tensorflow as tf
import tensorflow_hub as hub
from tf_agents.networks import network

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"


class HubPolicy(network.Network):
    def __init__(self, input_tensor_spec, action_spec, name="ActorNetwork"):
        super(HubPolicy, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        self.hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
        self.gru = tf.keras.layers.GRU(4, return_state=True)
        self.q_value_layer = tf.keras.layers.Dense(num_actions, activation=None)
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
        print("Entering call: ")
        # print(network_state)
        batch = []
        for i in range(observation.shape[0]):
            all_embeddings = []
            for index in range(self.number_of_strings):
                # print(observation[i, index])
                current_string = tf.reshape(observation[i, index], (1,))
                curr_embedding = self.hub_layer(current_string, training=training)
                curr_embedding = tf.expand_dims(curr_embedding, axis=0)
                all_embeddings.append(curr_embedding)
            all_embeddings = tf.concat(all_embeddings, axis=2)
            gru_output, network_state = self.gru(all_embeddings, initial_state=network_state)
            q_value = self.q_value_layer(gru_output, training=training)
            batch.append(q_value)
        q_values = tf.concat(q_value, axis=0)
        # print(q_value.shape)
        # print(network_state)
        return q_values, ()
