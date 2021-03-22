from typing import List

from torch.utils.tensorboard import SummaryWriter

from hearthstone.ladder.ladder import Contestant, update_ratings, print_standings
from hearthstone.simulator.replay.replay import Replay
from hearthstone.training.pytorch import tensorboard_altair
from hearthstone.training.pytorch.gae import GAEAnnotator
from hearthstone.training.pytorch.replay_buffer import EpochBuffer
from hearthstone.training.pytorch.surveillance import GlobalStepContext


class ReplaySink:
    def process(self, replay: Replay, learning_bot_contestant: Contestant, other_contestants: List[Contestant]):
        pass


class ExperiencePostProcessor(ReplaySink):
    def __init__(self,
                 epoch_buffer: EpochBuffer,
                 annotator: GAEAnnotator,
                 tensorboard: SummaryWriter,
                 global_step_context: GlobalStepContext
                 ):
        self.epoch_buffer = epoch_buffer
        self.annotator = annotator
        self.tensorboard = tensorboard
        self.global_step_context = global_step_context

    def process(self, replay: Replay, learning_bot_contestant: Contestant, other_contestants: List[Contestant]):
        self.annotator.annotate(replay)
        tensorboard_altair.plot_replay(replay, learning_bot_contestant.name, self.tensorboard,
                                       self.global_step_context)
        self._update_ratings(learning_bot_contestant, [learning_bot_contestant] + other_contestants, replay)
        self.epoch_buffer.add_replay(replay)

    @staticmethod
    def _update_ratings(learning_bot_contestant, all_contestants, replay):
        winner_names = replay.observer_annotations["RankingAnnotator"]
        final_boards = replay.observer_annotations["FinalBoardAnnotator"]
        print("---------------------------------------------------------------")
        print(winner_names)
        print("[" + ", ".join(final_boards[learning_bot_contestant.name]) + "]")
        ranked_contestants = sorted([c for c in all_contestants if c.name in winner_names],
                                    key=lambda c: winner_names.index(c.name))
        update_ratings(ranked_contestants)
        print_standings(all_contestants)
        for contestant in ranked_contestants:
            contestant.games_played += 1