# Straight up nicked from tf_agents lstm_encoding_network.py
# Need to mention copyright stuff etc. !

"""Keras LSTM Encoding Network.

Implements a network that will generate the following layers:

  [optional]: preprocessing_layers  # preprocessing_layers
  [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
  [optional]: Conv2D # input_conv_layer_params
  Flatten
  [optional]: Dense  # input_fc_layer_params
  [optional]: LSTM cell
  [optional]: Dense  # output_fc_layer_params
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

# MAX: Additional inputs
import tensorflow_hub as hub
EMBEDDING = "https://tfhub.dev/google/nnlm-en-dim50/2"
EMBEDDING_DIM = 50

KERAS_LSTM_FUSED = 2


@gin.configurable
class LSTMEmbeddingNetwork(network.Network):
    """Recurrent network with hard coded embedding input layer."""

    def __init__(
        self,
        input_tensor_spec,
        preprocessing_layers=None,
        preprocessing_combiner=None,
        conv_layer_params=None,
        input_fc_layer_params=(75, 40),
        lstm_size=None,
        output_fc_layer_params=(75, 40),
        activation_fn=tf.keras.activations.relu,
        rnn_construction_fn=None,
        rnn_construction_kwargs=None,
        dtype=tf.float32,
        name="LSTMEncodingNetwork",
    ):
        """Creates an instance of `LSTMEncodingNetwork`.

        Input preprocessing is possible via `preprocessing_layers` and
        `preprocessing_combiner` Layers.  If the `preprocessing_layers` nest is
        shallower than `input_tensor_spec`, then the layers will get the subnests.
        For example, if:

        ```python
        input_tensor_spec = ([TensorSpec(3)] * 2, [TensorSpec(3)] * 5)
        preprocessing_layers = (Layer1(), Layer2())
        ```

        then preprocessing will call:

        ```python
        preprocessed = [preprocessing_layers[0](observations[0]),
                        preprocessing_layers[1](observations[1])]
        ```

        However if

        ```python
        preprocessing_layers = ([Layer1() for _ in range(2)],
                                [Layer2() for _ in range(5)])
        ```

        then preprocessing will call:
        ```python
        preprocessed = [
          layer(obs) for layer, obs in zip(flatten(preprocessing_layers),
                                           flatten(observations))
        ]
        ```

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            observations.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations. All of these
            layers must not be already built.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them.  Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`. This
            layer must not be already built.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          input_fc_layer_params: Optional list of fully connected parameters, where
            each item is the number of units in the layer. These feed into the
            recurrent layer.
          lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
          output_fc_layer_params: Optional list of fully connected parameters, where
            each item is the number of units in the layer. These are applied on top
            of the recurrent layer.
          activation_fn: Activation function, e.g. tf.keras.activations.relu,.
          rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
            tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
            provide both rnn_construction_fn and lstm_size.
          rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
            rnn_construction_fn.

            The RNN will be constructed via:

            ```
            rnn_layer = rnn_construction_fn(**rnn_construction_kwargs)
            ```
          dtype: The dtype to use by the convolution, LSTM, and fully connected
            layers.
          name: A string representing name of the network.

        Raises:
          ValueError: If any of `preprocessing_layers` is already built.
          ValueError: If `preprocessing_combiner` is already built.
          ValueError: If neither `lstm_size` nor `rnn_construction_fn` are provided.
          ValueError: If both `lstm_size` and `rnn_construction_fn` are provided.
        """
        if lstm_size is None and rnn_construction_fn is None:
            raise ValueError(
                "Need to provide either custom rnn_construction_fn or " "lstm_size."
            )
        if lstm_size and rnn_construction_fn:
            raise ValueError(
                "Cannot provide both custom rnn_construction_fn and " "lstm_size."
            )

        kernel_initializer = tf.compat.v1.variance_scaling_initializer(
            scale=2.0, mode="fan_in", distribution="truncated_normal"
        )

        # MAX: Changed input_tensor_spec, as embedding before input_encoder
        input_encoder_spec = tf.TensorSpec(
            shape=(1, 2, 50), dtype=tf.dtypes.float32
        )
        input_encoder = encoding_network.EncodingNetwork(
            input_encoder_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=input_fc_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
        )

        # Create RNN cell
        # TODO: switch to GRUCells?
        if rnn_construction_fn:
            rnn_construction_kwargs = rnn_construction_kwargs or {}
            lstm_network = rnn_construction_fn(**rnn_construction_kwargs)
        else:
            if len(lstm_size) == 1:
                cell = tf.keras.layers.LSTMCell(
                    lstm_size[0], dtype=dtype, implementation=KERAS_LSTM_FUSED
                )
            else:
                cell = tf.keras.layers.StackedRNNCells(
                    [
                        tf.keras.layers.LSTMCell(
                            size, dtype=dtype, implementation=KERAS_LSTM_FUSED
                        )
                        for size in lstm_size
                    ]
                )
            lstm_network = dynamic_unroll_layer.DynamicUnroll(cell)

        output_encoder = []
        if output_fc_layer_params:
            output_encoder = [
                tf.keras.layers.Dense(
                    num_units,
                    activation=activation_fn,
                    kernel_initializer=kernel_initializer,
                    dtype=dtype,
                )
                for num_units in output_fc_layer_params
            ]

        counter = [-1]

        def create_spec(size):
            counter[0] += 1
            return tensor_spec.TensorSpec(
                size, dtype=dtype, name="network_state_%d" % counter[0]
            )

        state_spec = tf.nest.map_structure(create_spec, lstm_network.cell.state_size)

        super(LSTMEmbeddingNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=state_spec, name=name
        )

        self._conv_layer_params = conv_layer_params
        self._input_encoder = input_encoder
        self._lstm_network = lstm_network
        self._output_encoder = output_encoder

        # MAX: Added embedding layer
        # TODO: embedding trainable True or False?
        self._embedding = hub.KerasLayer(
            EMBEDDING, input_shape=[], dtype=tf.string, trainable=True
        )

    def call(self, observation, step_type, network_state=(), training=False):
        """Apply the network.

        Args:
          observation: A tuple of tensors matching `input_tensor_spec`.
          step_type: A tensor of `StepType.
          network_state: (optional.) The network state.
          training: Whether the output is being used for training.

        Returns:
          `(outputs, network_state)` - the network output and next network state.

        Raises:
          ValueError: If observation tensors lack outer `(batch,)` or
            `(batch, time)` axes.
        """
        num_outer_dims = nest_utils.get_outer_rank(observation, self.input_tensor_spec)
        if num_outer_dims not in (1, 2):
            raise ValueError(
                "Input observation must have a batch or batch x time outer shape."
            )

        # MAX: Use hard coded, custom embedding as first layer
        flattened_observation = tf.reshape(observation, (-1))
        embedded_observations = self._embedding(
            flattened_observation,
            training=training
        )
        embedded_observations = tf.reshape(
            embedded_observations,
            (observation.shape[0], observation.shape[1], EMBEDDING_DIM)
        )
        observation = embedded_observations

        has_time_dim = num_outer_dims == 2
        if not has_time_dim:
            # Add a time dimension to the inputs.
            observation = tf.nest.map_structure(
                lambda t: tf.expand_dims(t, 1), observation
            )
            step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1), step_type)

        state, _ = self._input_encoder(
            observation, step_type=step_type, network_state=(), training=training
        )

        network_kwargs = {}
        if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
            network_kwargs["reset_mask"] = tf.equal(
                step_type, time_step.StepType.FIRST, name="mask"
            )

        # Unroll over the time sequence.
        output = self._lstm_network(
            inputs=state,
            initial_state=network_state,
            training=training,
            **network_kwargs
        )

        if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
            state, network_state = output
        else:
            state = output[0]
            network_state = tf.nest.pack_sequence_as(
                self._lstm_network.cell.state_size, tf.nest.flatten(output[1:])
            )

        # FIXME: Using q value output encoder as in default QRNN Agent
        # FIXME: This should be changed to code from old hub_policy.py
        for layer in self._output_encoder:
            state = layer(state, training=training)

        # FIXME: Schmiert hier ab, weil already unsqueezed?
        # FIXME: Should this just be commented out???
        """
        if not has_time_dim:
            # Remove time dimension from the state.
            state = tf.squeeze(state, [1])
        """

        return state, network_state
