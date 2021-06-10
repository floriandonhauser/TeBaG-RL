from tf_agents.environments import tf_py_environment, utils

from environments import TWGameEnv


def test_environment_creation():
    """Test environment creation and validate with TF utils."""

    env_t, env_e = create_environments(debug=True)

    print("action_spec:", env_t.action_spec())
    print("time_step_spec.observation:", env_t.time_step_spec().observation)
    print("time_step_spec.step_type:", env_t.time_step_spec().step_type)
    print("time_step_spec.discount:", env_t.time_step_spec().discount)
    print("time_step_spec.reward:", env_t.time_step_spec().reward)


def create_environments(debug: bool = False):
    """Environment creation for test and evaluation

    Inspired by Noah's code.
    """

    env_name = "./resources/rewardsDense_goalBrief.ulx"
    path_verbs = "resources/words_verbs_short.txt"
    path_objs = "resources/words_objs_short.txt"
    path_badact = "./resources/bad_actions.txt"
    train_py_env = TWGameEnv(
        game_path=env_name,
        path_verb=path_verbs,
        path_obj=path_objs,
        path_badact=path_badact,
        debug=debug,
    )
    eval_py_env = TWGameEnv(
        game_path=env_name,
        path_verb=path_verbs,
        path_obj=path_objs,
        path_badact=path_badact,
        debug=debug,
    )

    if debug:
        utils.validate_py_environment(train_py_env, episodes=5)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    return train_env, eval_env
