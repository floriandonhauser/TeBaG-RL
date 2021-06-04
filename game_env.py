"""Game environment class for project based on TextWorld"""

import textworld
import textworld.gym
import gym

from dummy_agend import DummyAgent


class GameEnv:
    """Game environment class to play TextWorld games with an agent giving commands.

    Parameters:
    -----------
    agent
        Agent to call commands via agent.get_cmd()
    game_path: str
        Path to game file (ulx. only?)
    max_step, no_games: [4, 2]
        Maximum number of steps tried per game and number of games to be played
    debug: True
        Turning on/off printing of states, scores, etc.
    """

    def __init__(
        self,
        agent,
        game_path: str,
        max_step: int = 4,
        no_games: int = 2,
        debug: bool = True,
    ):
        self.agent = agent
        self.game_path = game_path
        self.max_step = max_step
        self.no_games = no_games
        self.debug = debug

        self.curr_env = None
        self.curr_env_state = None

    def _start_game(self):
        """Initializing game environment"""

        if self.debug:
            print("Starting new game.")

        request_infos = textworld.EnvInfos(
            description=True,
            admissible_commands=True,
            entities=True,
            inventory=True,
            won=True,
            lost=True,
        )
        env_id = textworld.gym.register_game(self.game_path, request_infos)
        self.curr_env = gym.make(env_id)
        init_obs, init_info = self.curr_env.reset()

        return init_obs, init_info

    def _end_game(self):
        """Ending game environment"""

        if self.debug:
            print("Ending current game.")
        self.curr_env.close()

    def _print_current(self):
        """Print current output from game. Placeholder function."""

        print("Observation: \n", self.curr_env_state[0])
        print("Description: \n", self.curr_env_state[3]["description"])
        print("Inventory: \n", self.curr_env_state[3]["inventory"])
        print("Score: \t", self.curr_env_state[1])

    def play(self):
        """Game play loop for given number of tries and games."""

        for i in range(self.no_games):
            obs, infos = self._start_game()
            self.curr_env_state = self.curr_env.step("look")

            for step in range(self.max_step):
                if (
                    self.curr_env_state[2]
                    or self.curr_env_state[3]["won"]
                    or self.curr_env_state[3]["lost"]
                ):
                    break

                cmd = self.agent.get_cmd(self.curr_env_state)
                if self.debug:
                    print(f"Doing: {cmd}")
                self.curr_env_state = self.curr_env.step(cmd)

                if self.debug:
                    self._print_current()

            self._end_game()


if __name__ == "__main__":
    d_agend = DummyAgent()
    path = "/home/max/Software/TextWorld/notebooks/games/rewardsDense_goalBrief.ulx"

    g_env = GameEnv(d_agend, path)
    g_env.play()
