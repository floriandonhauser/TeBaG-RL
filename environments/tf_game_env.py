"""TF python environment class for game based on TextWorld"""

from abc import ABC

import gym
import numpy as np
import textworld
import textworld.gym
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

STR_TYPE = "S500"


class TWGameEnv(py_environment.PyEnvironment, ABC):
    """Game environment for TextWorld games in TensorFlow agents.

    Parameters:
    -----------
    game_path: str
        Path to game file
    path_verb, path_obj: str
        Path to verb and object files to create commands as VERB + OBJ
    path_badact: str
        Path to list of bad environment observation returns from nonsense commands.
    debug: True
        Turning on/off printing of states, commands, etc.
    flatten_actspec: False
        Flattening action space from 2D (ver, obj) to list of all possible combinations for 1D action space.
    """

    def __init__(self, game_path: str, path_verb: str, path_obj: str, path_badact: str, debug: bool = False, flatten_actspec: bool = False):
        self._game_path = game_path
        self._path_verb = path_verb
        self._path_obj = path_obj
        self._path_badact = path_badact
        self._debug = debug
        self._flatten_actspec = flatten_actspec

        self._list_verb = self._get_words(self._path_verb)
        self._list_obj = self._get_words(self._path_obj)
        self._list_badact = self._get_words(self._path_badact)

        if self._flatten_actspec:
            self._list_verbobj = [v + " " + o for v in self._list_verb for o in self._list_obj]
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(1,),
                dtype=np.uint16,
                minimum=0,
                maximum=(len(self._list_verbobj) - 1),
                name="action",
            )

        else:
            self._list_verbobj = None
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(2,),
                dtype=np.uint16,
                minimum=[0, 0],
                maximum=[len(self._list_verb) - 1, len(self._list_obj) - 1],
                name="action",
            )

        self._observation_spec = array_spec.ArraySpec(shape=(2,), dtype=STR_TYPE, name="observation")

        self.curr_TWGym = None
        self._state = None
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        self._episode_ended = False
        self._start_game()

        return ts.restart(self._conv_pass_state(self._state))

    def _step(self, action):
        if self._episode_ended:
            # Last action ended episode. Ignore current action and start new episode.
            return self.reset()

        if self._state["done"] or self._state["won"] or self._state["lost"]:
            self._episode_ended = True

        old_state = self._state
        cmd = self._conv_to_cmd(action)
        self._state = self._conv_to_state(*self.curr_TWGym.step(cmd))
        new_state = self._state

        if self._debug:
            print(self._state)

        # TODO: adjust reward and discount
        pass_state = self._conv_pass_state(self._state)
        reward = self._calc_reward(new_state, old_state, cmd)
        if self._debug:
            print(f"Reward = {reward}")

        if self._episode_ended:
            return ts.termination(pass_state, reward)
        else:
            return ts.transition(pass_state, reward=reward, discount=1.0)

    def _start_game(self):
        """Initializing new game environment in TextWorld"""

        if self._debug:
            print("Starting new game.")

        request_info = textworld.EnvInfos(
            description=True,
            admissible_commands=True,
            entities=True,
            inventory=True,
            won=True,
            lost=True,
        )
        env_id = textworld.gym.register_game(self._game_path, request_info)
        self.curr_TWGym = gym.make(env_id)
        self.curr_TWGym.reset()

        self._state = self._conv_to_state(*self.curr_TWGym.step("look"))
        if self._debug:
            print(self._state)

    def _conv_to_cmd(self, action_ind: list):
        """Convert indices from agent into string command via imported files."""

        if self._flatten_actspec:
            cmd_str = self._list_verbobj[action_ind]
        else:
            verb = self._list_verb[action_ind[0]]
            # EMTPY obj should be empty string
            if action_ind[1] == 0:
                obj = ""
            else:
                obj = self._list_obj[action_ind[1]]
            cmd_str = verb + " " + obj
        if self._debug:
            print(f"Doing: {cmd_str}")

        return cmd_str

    def _calc_reward(self, new_state, old_state, cmd):
        """Calculate reward based on different environment returns and changes."""

        reward = 0

        # Use score difference as base reward
        reward += new_state["score"] - old_state["score"]

        # Use change in environment description to reward changes
        # Todo: Maybe use length check? If strings do not saturate length, it could work
        inv_change = False if new_state["inventory"] == old_state["inventory"] else True
        des_change = False if new_state["description"] == old_state["description"] else True
        if inv_change or des_change:
            reward += 1
        else:
            reward -= 1

        # Punish useless actions
        if np.array([elem in new_state["obs"] for elem in self._list_badact]).sum():
            reward -= 1

        # TODO: Check if command was partly in admissible commands (right verb)
        # TODO: Include "Win" or "Lost" state

        return reward

    @staticmethod
    def _conv_to_state(obs: str, score: int, done: bool, info: dict) -> np.array:
        """Convert TextWorld gym env output into nested array"""

        # TODO: Pre-processing text?
        return {
            "score": score,
            "done": done,
            "won": info["won"],
            "lost": info["lost"],
            "obs": obs,
            "description": info["description"],
            "inventory": info["inventory"],
            "admissible_commands": info["admissible_commands"],
            "entities": info["entities"],
        }

    @staticmethod
    def _conv_pass_state(state):
        """Select information to pass from current state and create app. np.array."""
        return np.array([state["description"], state["inventory"]], dtype=STR_TYPE)

    @staticmethod
    def _get_words(path: str):
        """Import words (verbs or objects) from verb txt file"""

        with open(path, "r") as f:
            content = [item.strip() for item in f]
        return content
