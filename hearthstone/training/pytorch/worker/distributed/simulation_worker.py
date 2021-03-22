import random
import threading
import time
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

import torch
from torch.distributed import rpc

from hearthstone.ladder.ladder import Contestant
from hearthstone.simulator.host.round_robin_host import RoundRobinHost
from hearthstone.simulator.replay.annotators.final_board_annotator import FinalBoardAnnotator
from hearthstone.simulator.replay.annotators.ranking_annotator import RankingAnnotator
from hearthstone.training.pytorch.gae import GAEAnnotator
from hearthstone.training.pytorch.tensorboard_altair import TensorboardAltairAnnotator


class SimulationWorker:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.thread_pool = ThreadPoolExecutor()
        torch.set_num_threads(1)

    def play_game(self,
                  learning_bot_contestant: Contestant,
                  other_contestants: List[Contestant],
                  game_size: int):
        with torch.no_grad():
            round_contestants = [learning_bot_contestant] + random.sample(other_contestants,
                                                                          k=game_size - 1)
            host = RoundRobinHost(
                {contestant.name: contestant.agent_generator() for contestant in round_contestants},
                [RankingAnnotator(),
                 FinalBoardAnnotator(),
                 TensorboardAltairAnnotator([learning_bot_contestant.name])]
            )
            start = time.time()
            host.play_game()
            print(f"Worker played 1 game. Time taken: {time.time() - start} seconds.")
            replay = host.get_replay()
            return replay