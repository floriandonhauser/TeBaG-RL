from typing import Mapping, Any, Optional
import tensorflow as tf

class Agent(tf.keras.Model):
    """
    This class is an example agent used to play the game
    """
    def __init__(self, input_size: int):
        super(Agent, self).__init__()

    def call(self, inputs, **kwargs):
        """
        Method used in play to confirm to tf.keras.Model

        Parameters:
        inputs: Input vector

        Returns:
        scores for the possible actions
        """
        scores = None
        return scores

    def reset_hidden(self, batch_size: int):
        """
        This method is used to reset the hidden state of the model
        """
        pass

    def act(self, state: Mapping[str, Any]) -> Optional[str]:
        """
        This method is used to find the action that should be performed in the current state

        Parameters:
        state: Dictionary of strings describing the state

        Returns:
        action that should be taken
        """
        action = None
        return action

    def save_network(self):
        """
        This method can be used to safe the model's weights on disk
        """
        pass

    def load_network(self):
        """
        This method can be used to load the model's weights from disk
        """
        pass