"""Module for centralized environment creation"""

from tf_agents.environments import tf_py_environment, utils
from environments import TWGameEnv


def create_environments(debug: bool = False, flatten_actspec: bool = True):
    """Environment creation for test and evaluation.
    """

    env_name = "./resources/rewardsDense_goalBrief.ulx"
    path_verbs = "./resources/words_verbs_short.txt"
    path_objs = "./resources/words_objs_short.txt"
    path_badact = "./resources/bad_actions.txt"

    train_py_env = TWGameEnv(
        game_path=env_name,
        path_verb=path_verbs,
        path_obj=path_objs,
        path_badact=path_badact,
        debug=debug,
        flatten_actspec=flatten_actspec,
    )
    eval_py_env = TWGameEnv(
        game_path=env_name,
        path_verb=path_verbs,
        path_obj=path_objs,
        path_badact=path_badact,
        debug=debug,
        flatten_actspec=flatten_actspec,
    )

    if debug:
        utils.validate_py_environment(train_py_env, episodes=5)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    return train_env, eval_env