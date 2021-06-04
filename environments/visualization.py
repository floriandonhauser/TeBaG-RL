import tensorflow as tf


def get_fit_callback():
    """
    Returns a callback that can be used during fit to visualize the loss
    """
    return tf.keras.callbacks.TensorBoard(log_dir="tensorboard_visualizations", histogram_freq=1)
