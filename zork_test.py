"""
Testing Zork/Python interaction time

max.lamparth@tum.de

Needs Zork installed and callable as "zork" in the terminal.

Results:
-------
Taken time for 10 commands: 0.9070s
Time per commands 0.0907s
Sleep time used to communicate: 0.03

Notes:
------
    Used commands:
       The 'INVENTORY' command lists the objects in your possession.
       The 'LOOK' command prints a description of your surroundings.
       Random action for testing
    Unused commands:
       The 'SCORE' command prints your current score and ranking.
       The 'TIME' command tells you how long you have been playing.
       The 'DIAGNOSE' command reports on your injuries, if any.

    IO nicked from
    https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
"""

from subprocess import PIPE, Popen
from time import sleep, time
from queue import Queue, Empty
from threading import Thread


class ZorkGame:
    """Class for running and interacting with the game."""

    def __init__(self):
        self._p = None
        self._q = None
        self._t = None
        self._sleep_time = 0.03

    def __call__(self, n_trials: int = 10):
        """Run fixed testing"""

        self._start_game()
        self._print_current()

        start_time = time()
        # Simulate one cycle/round
        for i in range(n_trials):
            # Look around (update location state)
            self._do_thing(command="L")
            self._print_current()

            # Check inventory (update inventory)
            self._do_thing(command="I")
            self._print_current()

            # Do an action (agend output)
            self._do_thing(command="Bollocks")
            self._print_current()
        end_time = time()

        print(f"\nTaken time for {n_trials} commands: {end_time - start_time:0.4f}s")
        print(f"Time per commands {(end_time - start_time) / n_trials:0.4f}s")
        print(f"Sleep time used to communicate: {self._sleep_time}")

        self._end_game()

    @staticmethod
    def _enqueue_output(out, queue):
        """Get terminal output from game"""

        for line in iter(out.readline, b""):
            queue.put(line)
        out.close()

    def _start_game(self):
        """Initiate a game of Zork"""

        print("Starting game")
        self._p = Popen("zork", stdin=PIPE, stdout=PIPE, stderr=PIPE)

        self._q = Queue()
        self._t = Thread(target=self._enqueue_output, args=(self._p.stdout, self._q))
        self._t.daemon = True
        self._t.start()
        sleep(self._sleep_time)

    def _end_game(self):
        """End the game"""

        print("Ending game")
        self._p.terminate()
        self._p.kill()

    def _do_thing(self, command: str):
        """Executing a command"""

        print(f"Do '{command}'")
        self._p.stdin.write(bytes(command + "\n", "ascii"))
        self._p.stdin.flush()
        sleep(self._sleep_time)

    def _print_current(self):
        """Print current output from game. Placeholder function."""

        not_empty = True
        while not_empty:
            try:
                line = self._q.get_nowait()
            except Empty:
                not_empty = False
            else:
                print(line.strip())


zg = ZorkGame()
# Run test
zg()
