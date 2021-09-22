import random
import time
from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter

from hearthstone.ladder.ladder import Contestant, update_ratings, print_standings
from hearthstone.simulator.host.round_robin_host import RoundRobinHost
from hearthstone.training.pytorch.gae import GAEAnnotator
from hearthstone.training.pytorch.replay_buffer import EpochBuffer
from hearthstone.training.pytorch.surveillance import GlobalStepContext
from hearthstone.training.pytorch.tensorboard_altair import TensorboardAltairAnnotator, plot_replay


class SingleWorkerPool:
    def __init__(self,
                 epoch_buffer: EpochBuffer,
                 annotator: GAEAnnotator,
                 tensorboard: SummaryWriter,
                 global_step_context: GlobalStepContext):
        """
        Worker is responsible for setting up games where the learning bot plays against a random set of opponents.
        """
        self.epoch_buffer: EpochBuffer = epoch_buffer
        self.annotator: GAEAnnotator = annotator
        self.tensorboard = tensorboard
        self.global_step_context = global_step_context

    def play_games(self, learning_bot_contestant: Contestant,
                   other_contestants: List[Contestant],
                   game_size: int):
        round_contestants = [learning_bot_contestant] + random.sample(other_contestants,
                                                                      k=game_size - 1)
        with torch.no_grad():
            host = RoundRobinHost(
                {contestant.name: contestant.agent_generator() for contestant in round_contestants},
                [TensorboardAltairAnnotator([learning_bot_contestant.name])]
            )
            start = time.time()
            host.play_game()
            print(f"Worker played 1 game. Time taken: {time.time() - start} seconds.")
            winner_names = list(reversed([name for name, player in host.tavern.losers]))
            print("---------------------------------------------------------------")
            print(winner_names)
            print(host.tavern.players[learning_bot_contestant.name].in_play,
                  host.tavern.players[learning_bot_contestant.name].hero)
            ranked_contestants = sorted(round_contestants, key=lambda c: winner_names.index(c.name))
            update_ratings(ranked_contestants)
            print_standings([learning_bot_contestant] + other_contestants)
            for contestant in round_contestants:
                contestant.games_played += 1

            replay = host.get_replay()
            self.annotator.annotate(replay)
            plot_replay(replay, learning_bot_contestant.name, self.tensorboard, self.global_step_context)
            self.epoch_buffer.add_replay(replay)
