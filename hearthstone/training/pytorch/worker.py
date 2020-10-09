import random
from typing import List

from torch.utils.tensorboard import SummaryWriter

from hearthstone.ladder.ladder import Contestant, update_ratings, print_standings
from hearthstone.simulator.host.round_robin_host import RoundRobinHost
from hearthstone.training.pytorch.gae import GAEAnnotator
from hearthstone.training.pytorch.replay_buffer import EpochBuffer
from hearthstone.training.pytorch.surveillance import GlobalStepContext
from hearthstone.training.pytorch.tensorboard_altair import TensorboardAltairAnnotator, plot_replay


class Worker:
    def __init__(self, learning_bot_contestant: Contestant,
                 other_contestants: List[Contestant],
                 game_size: int,
                 epoch_buffer: EpochBuffer,
                 annotator: GAEAnnotator,
                 tensorboard: SummaryWriter,
                 global_step_context: GlobalStepContext):
        """
        Worker is responsible for setting up games where the learning bot plays against a random set of opponents.

        Args:
            learning_bot_contestant (Contestant):
            other_contestants (List[Contestant]):
        """
        self.other_contestants = other_contestants
        self.learning_bot_contestant = learning_bot_contestant
        self.game_size = game_size
        self.epoch_buffer: EpochBuffer = epoch_buffer
        self.annotator: GAEAnnotator = annotator
        self.tensorboard = tensorboard
        self.global_step_context = global_step_context

    def play_game(self):
        round_contestants = [self.learning_bot_contestant] + random.sample(self.other_contestants,
                                                                           k=self.game_size - 1)
        host = RoundRobinHost(
            {contestant.name: contestant.agent_generator() for contestant in round_contestants},
            [TensorboardAltairAnnotator([self.learning_bot_contestant.name])]
        )
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

        replay = host.get_replay()
        self.annotator.annotate(replay)
        plot_replay(replay, self.learning_bot_contestant.name, self.tensorboard, self.global_step_context)
        self.epoch_buffer.add_replay(replay)
