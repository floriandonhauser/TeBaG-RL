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
    reward_dict: dict = None,
    onlytrain: bool = False,
):
    """Central environment creation method.

    Parameters:
    -----------
    debug: bool
        Enables debug outputs and tests environment with tf_agent util method.
    flatten_actspec: bool
        Flatten action spec of environment by multiplying all verbs with all objects.
        Action spec will be 1D instead of 2D.
    expand_vocab: bool
        Activate expand_vocab mode. If true, environment will search for entities in
        TextWorld returns to be on object vocabulary. Missing entries will be appended
        to object vocab file. As this changes dimension of the action spec, agents would
        need to be retrained.
    no_episodes: int
        Number of episodes to test environment in debug mode.
    reward_dict: dict
        Dictionary with values to reward/punish agent for certain state changes.
    onlytrain: bool
        Toggle creation of train and eval environment at the same time (optimization.)
    env_name, path_verbs, path_objs, path_badact: str
        Paths to game file, verb vocab file, object vocab file and bad action
        observation returns.

    """

    train_py_env = TWGameEnv(
        game_path=env_name,
        path_verb=path_verbs,
        path_obj=path_objs,
        path_badact=path_badact,
        debug=debug,
        flatten_actspec=flatten_actspec,
        expand_vocab=expand_vocab,
        reward_dict=reward_dict,
    )
    if not onlytrain:
        eval_py_env = TWGameEnv(
            game_path=env_name,
            path_verb=path_verbs,
            path_obj=path_objs,
            path_badact=path_badact,
            debug=debug,
            flatten_actspec=flatten_actspec,
            reward_dict=reward_dict,
        )

    if debug:
        utils.validate_py_environment(train_py_env, episodes=no_episodes)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    if not onlytrain:
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    else:
        eval_env = None

    return train_env, eval_env, train_py_env.num_verb, train_py_env.num_obj


if __name__ == "__main__":
    create_environments(debug=True)
