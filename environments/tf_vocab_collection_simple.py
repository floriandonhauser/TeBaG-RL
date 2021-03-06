"""Module for automatic object vocabulary collection from train games."""

import os
from glob import glob
from environments import create_environments


def run_auto_vocab():
    """"""

    path = "../resources/train_games_lvl2/"
    gamefiles = glob(os.path.join(path, "*.ulx"))
    for game in gamefiles:
        create_environments(env_name=game, debug=True, expand_vocab=True, no_episodes=1)


if __name__ == "__main__":
    run_auto_vocab()
