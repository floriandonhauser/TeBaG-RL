"""Module for environment testing."""

from environments import create_environments


def test_environment_creation():
    """Test environment creation and validate with TF utils."""

    env_t, env_e, _, _ = create_environments(debug=True)

    print("action_spec:", env_t.action_spec())
    print("time_step_spec.observation:", env_t.time_step_spec().observation)
    print("time_step_spec.step_type:", env_t.time_step_spec().step_type)
    print("time_step_spec.discount:", env_t.time_step_spec().discount)
    print("time_step_spec.reward:", env_t.time_step_spec().reward)
