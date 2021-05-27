""""""


class DummyAgent:
    """"""

    def __init__(self, def_cmd: str = "look"):
        self.cmd = def_cmd

    def get_cmd(self, env_input):
        """"""
        return self.cmd
