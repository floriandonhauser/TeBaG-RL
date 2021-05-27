"""Skeletal game environment example to use with TextWorld using gym."""

import textworld.gym
import gym

# Init
path = "/home/max/Software/TextWorld/notebooks/games/rewardsDense_goalBrief.ulx"
request_infos = textworld.EnvInfos(
    description=True,
    admissible_commands=True,
    entities=True,
    inventory=True,
    won=True,
    lost=True,
)

env_id = textworld.gym.register_game(path, request_infos)
env = gym.make(env_id)

obs, infos = env.reset()
print(env.reset())
print(obs)
print(infos)

# Loop
finished = False
while not finished:
    command = input()

    obs, score, done, infos = env.step(command)

    print("obs\t", obs)
    print("desc\t", infos["description"])
    # print(done)
    # print(score)

    if infos["won"] or infos["lost"] or done:
        env.close()
        finished = True
