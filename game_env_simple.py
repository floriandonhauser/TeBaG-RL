"""Simplest game environment example to use with TextWorld."""

import textworld

path = "/home/max/Software/TextWorld/notebooks/games/rewardsDense_goalBrief.ulx"

# Init
env = textworld.start(path)
env.reset()
env.seed(seed=None)

# Loop
while True:
    env.render()
    command = input()

    game_state, reward, done = env.step(command)

    print(game_state.feedback)
    print(game_state.description)
    print(game_state.inventory)
    print(game_state.score)
    print(game_state.moves)
    print(game_state.score)
