import typing

from hearthstone.simulator.replay.replay import Replay

if typing.TYPE_CHECKING:
    pass


class Host:
    def start_game(self):
        pass

    def play_round(self):
        pass

    def game_over(self):
        pass

    def play_game(self):
        pass

    def get_replay(self) -> Replay:
        pass


