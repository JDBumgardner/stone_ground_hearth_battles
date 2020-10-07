from random import random
from typing import List

from hearthstone.ladder.ladder import Contestant, update_ratings, print_standings
from hearthstone.simulator.host import RoundRobinHost, Host
from hearthstone.training.pytorch.replay_buffer import EpochBuffer


class Worker:
    def __init__(self, learning_bot_contestant: Contestant, other_contestants: List[Contestant], game_size: int, epoch_buffer: EpochBuffer):
        """
        Worker is responsible for setting up games where the learning bot plays against a random set of opponents and
        provides a way to step through the games one action at a time.

        Args:
            learning_bot_contestant (Contestant):
            other_contestants (List[Contestant]):
        """
        self.other_contestants = other_contestants
        self.learning_bot_contestant = learning_bot_contestant
        self.game_size = game_size
        self.epoch_buffer: EpochBuffer = epoch_buffer

    def play_game(self):
        round_contestants = [self.learning_bot_contestant] + random.sample(self.other_contestants,
                                                                           k=self.game_size - 1)
        host = RoundRobinHost(
            {contestant.name: contestant.agent_generator() for contestant in round_contestants})
        host.play_game()
        winner_names = list(reversed([name for name, player in host.tavern.losers]))
        print("---------------------------------------------------------------")
        print(winner_names)
        print(host.tavern.players[self.learning_bot_contestant.name].in_play)
        ranked_contestants = sorted(round_contestants, key=lambda c: winner_names.index(c.name))
        update_ratings(ranked_contestants)
        print_standings([self.learning_bot_contestant] + self.other_contestants)
        for contestant in round_contestants:
            contestant.games_played += 1