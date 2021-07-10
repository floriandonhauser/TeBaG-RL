"""Module for centralized environment creation"""

from tf_agents.environments import tf_py_environment, utils

from environments import TWGameEnv
from resources import DEFAULT_PATHS


def create_environments(
    debug: bool = False,
    flatten_actspec: bool = True,
    expand_vocab: bool = False,
    no_episodes: int = 5,
    env_name: str = DEFAULT_PATHS["env_name"],
    path_verbs: str = DEFAULT_PATHS["path_verbs"],
    path_objs: str = DEFAULT_PATHS["path_objs"],
    path_badact: str = DEFAULT_PATHS["path_badact"],
):
    """Environment creation for test and evaluation."""

    train_py_env = TWGameEnv(
        game_path=env_name,
        path_verb=path_verbs,
        path_obj=path_objs,
        path_badact=path_badact,
        debug=debug,
        flatten_actspec=flatten_actspec,
        expand_vocab=expand_vocab,
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
        utils.validate_py_environment(train_py_env, episodes=no_episodes)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    return train_env, eval_env, train_py_env.num_verb, train_py_env.num_obj


if __name__ == "__main__":
    create_environments(debug=True)
