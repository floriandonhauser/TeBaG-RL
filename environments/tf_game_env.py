"""TF python environment class for game based on TextWorld"""

from abc import ABC

import gym
import numpy as np
import textworld
import textworld.gym
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# defaults
STR_TYPE = "S500"
HASH_LIST_LENGTH = 10
# all positive int!
REWARD_DICT = {
    "win_lose_value": 100,
    "max_loop_pun": 5,
    "change_reward": 1,
    "useless_act_pun": 1,
    "verb_in_adm": 1,
}


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
        Flattening action space from 2D (ver, obj) to list of all possible combinations
        for 1D action space.
    expand_vocab: False
        Turn on automatic object vocabulary expansion.
    reward_dict: dict
        Dictionary with values to reward/punish agent for certain state changes.
        REWARD_DICT = {
            "win_lose_value": +- Reward for winning/losing game
            "max_loop_pun": - Reward for looping over same states
            "change_reward": + Reward for changing surroundings or inventory
            "useless_act_pun": - Reward for using non-recognizable action
            "verb_in_adm": + Reward for using a verb that is at least admissible
        }
    hash_list_length: int
        Length of last states to be hashed for change-of-state comparisons and reward
        calculations.
    obs_stype: str
        Type of observation string to be cast. Needs to be numpy type.
    """

    def __init__(
        self,
        game_path: str,
        path_verb: str,
        path_obj: str,
        path_badact: str,
        debug: bool = False,
        flatten_actspec: bool = False,
        expand_vocab: bool = False,
        reward_dict: dict = REWARD_DICT,
        hash_list_length: int = HASH_LIST_LENGTH,
        obs_stype: str = STR_TYPE,
    ):
        self._game_path = game_path
        self._path_verb = path_verb
        self._path_obj = path_obj
        self._path_badact = path_badact
        self._bool_dict = {
            "debug": debug,
            "flatten_actspec": flatten_actspec,
            "expand_vocab": expand_vocab,
        }
        if reward_dict is not None:
            self._reward_dict = reward_dict
        else:
            self._reward_dict = REWARD_DICT
        self._hash_list_length = hash_list_length
        self._obs_stype = obs_stype

        self._list_verb = self._get_words(self._path_verb)
        self._list_obj = self._get_words(self._path_obj)
        self._list_badact = self._get_words(self._path_badact)
        self._missing_obj = []

        self.num_verb = len(self._list_verb)
        self.num_obj = len(self._list_obj)

        if self._bool_dict["flatten_actspec"]:
            # TODO: First obj is EMPTY and should not be printed
            self._list_verbobj = [
                v + " " + o for v in self._list_verb for o in self._list_obj
            ]
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=(len(self._list_verbobj) - 1),
                name="action",
            )

        else:
            self._list_verbobj = None
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(2,),
                dtype=np.int32,
                minimum=[0, 0],
                maximum=[len(self._list_verb) - 1, len(self._list_obj) - 1],
                name="action",
            )

        self._observation_spec = array_spec.ArraySpec(
            shape=(2,), dtype=self._obs_stype, name="observation"
        )

        self._hash_dsc = [0] * self._hash_list_length
        self._hash_inv = [0] * self._hash_list_length

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
        self._update_hash_cache(new_state)

        if self._bool_dict["debug"]:
            print(self._state)

        if self._bool_dict["expand_vocab"]:
            for ent in new_state["entities"]:
                if ent.find(" ") == -1:
                    sub_str = ent
                else:
                    sub_str = ent[ent.rfind(" ") + 1 :]

                if sub_str not in self._missing_obj:
                    found_obj = self._find_word_in_list(
                        word_str=sub_str, word_list=self._list_obj
                    )
                    if not found_obj:
                        self._missing_obj.append(sub_str)

        # TODO: adjust discount in tf_agents.trajectories.time_step.transition?
        pass_state = self._conv_pass_state(self._state)
        reward = self._calc_reward(new_state, old_state, cmd)
        if self._bool_dict["debug"]:
            print(f"Reward = {reward}")

        if self._episode_ended:
            if self._bool_dict["expand_vocab"]:
                for new_word in self._missing_obj:
                    self._append_word_to_file(word=new_word, file=self._path_obj)

            return ts.termination(pass_state, reward)
        else:
            return ts.transition(pass_state, reward=reward, discount=1.0)

    def _start_game(self):
        """Initializing new game environment in TextWorld"""

        if self._bool_dict["debug"]:
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
        if self._bool_dict["debug"]:
            print(self._state)

    def _conv_to_cmd(self, action_ind: list):
        """Convert indices from agent into string command via imported files."""

        if self._bool_dict["flatten_actspec"]:
            cmd_str = self._list_verbobj[action_ind]
        else:
            verb = self._list_verb[action_ind[0]]
            # EMTPY obj should be empty string
            if action_ind[1] == 0:
                obj = ""
            else:
                obj = self._list_obj[action_ind[1]]
            cmd_str = verb + " " + obj
        if self._bool_dict["debug"]:
            print(f"Doing: {cmd_str}")

        return cmd_str

    def _calc_reward(self, new_state, old_state, cmd):
        """Calculate reward based on different environment returns and changes."""

        reward = 0

        # Use score difference as base reward
        reward += new_state["score"] - old_state["score"]

        # Punish useless actions from know game return statements
        if np.array([elem in new_state["obs"] for elem in self._list_badact]).sum():
            reward -= self._reward_dict["useless_act_pun"]

        # Use change in environment description to reward changes
        inv_change = self._calc_cache_changes(self._hash_inv)
        des_change = self._calc_cache_changes(self._hash_dsc)
        if inv_change <= 1 or des_change <= 1:
            reward += self._reward_dict["change_reward"]
        else:
            # at least 1, at max self._reward_dict["max_loop_pun"]
            reward -= min(
                [inv_change - 1, des_change - 1, self._reward_dict["max_loop_pun"]]
            )

        # Greatly reward/punish win/lose of game
        if new_state["won"]:
            print("GAME WON")
            reward += self._reward_dict["win_lose_value"]
        elif new_state["lost"]:
            print("GAME LOST")
            reward -= self._reward_dict["win_lose_value"]

        # Check if verb in command was in admissible commands
        cmd_in_adm = self._find_word_in_list(
            word_str=cmd[: cmd.find(" ")], word_list=new_state["admissible_commands"]
        )
        if cmd_in_adm:
            reward += self._reward_dict["verb_in_adm"]

        return reward

    def _update_hash_cache(self, curr_state):
        """Use new state to add current desc and inv to hashed list of last states."""

        # Advanced hashing with import hashlib
        self._hash_dsc.append(hash(curr_state["description"]))
        self._hash_dsc.pop(0)
        self._hash_inv.append(hash(curr_state["inventory"]))
        self._hash_inv.pop(0)

    def _conv_pass_state(self, state):
        """Select information to pass from current state and create app. np.array."""
        return np.array(
            [state["description"], state["inventory"]], dtype=self._obs_stype
        )

    @staticmethod
    def _append_word_to_file(word: str, file: str):
        """Helper method to append word to a file for vocab generation"""

        with open(file, "a+") as file_object:
            file_object.seek(0)
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("\n")
            file_object.write(word)

    @staticmethod
    def _find_word_in_list(word_str: str, word_list: list) -> bool:
        """Find whether a substring is in a list of longer strings"""

        count = np.asarray([word_str in adm for adm in word_list]).sum()
        if count >= 1:
            return True
        else:
            return False

    @staticmethod
    def _calc_cache_changes(cache: list) -> int:
        """Sum over how many times latest state was in cache"""
        return (np.asarray(cache) == cache[-1]).sum()

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
    def _get_words(path: str):
        """Import words (verbs or objects) from verb txt file"""

        with open(path, "r") as f:
            content = [item.strip() for item in f]
        return content
