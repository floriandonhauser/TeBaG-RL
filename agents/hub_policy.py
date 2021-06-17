import tensorflow as tf
import tensorflow_hub as hub
from tf_agents.networks import network

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"


class HubPolicy(network.Network):
    def __init__(self, input_tensor_spec, action_spec, num_verb, num_obj, name="ActorNetwork"):
        super(HubPolicy, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        assert num_actions == num_verb*num_obj
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
        print("Entering call: ")
        # print(network_state)
        print("observation.shape", observation.shape)
        print(observation)
        batch = []
        for i in range(observation.shape[0]):
            all_embeddings = []
            for index in range(self.number_of_strings):
                # print(observation[i, index])
                current_string = tf.reshape(observation[i, index], (1,))
                print("current_string.shape", current_string.shape)
                print("training", training)
                print(current_string)
                curr_embedding = self.hub_layer(current_string, training=training)
                print("curr_embedding.shape 1", curr_embedding.shape)
                curr_embedding = tf.expand_dims(curr_embedding, axis=0)
                print("curr_embedding.shape 2", curr_embedding.shape)
                all_embeddings.append(curr_embedding)
            all_embeddings = tf.concat(all_embeddings, axis=2)
            print("all_embeddings.shape", all_embeddings.shape)
            gru_output, network_state = self.gru(all_embeddings, initial_state=network_state)
            verb_q_value = tf.expand_dims(self.verb_layer(gru_output, training=training), axis=1)
            obj_q_value = tf.expand_dims(self.obj_layer(gru_output, training=training), axis=1)
            q_value_multiplied = tf.matmul(verb_q_value, obj_q_value, transpose_a=True)
            print("--------------------------\n", verb_q_value, obj_q_value, q_value_multiplied, "\n------------------------")
            q_value_multiplied = tf.reshape(q_value_multiplied, (q_value_multiplied.shape[0], -1))
            batch.append(q_value_multiplied)
        q_values = tf.concat(batch, axis=0)
        # print(q_value.shape)
        # print(network_state)
        return q_values, ()
