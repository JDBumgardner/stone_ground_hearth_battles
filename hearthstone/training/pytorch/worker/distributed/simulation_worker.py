import asyncio
import random
import time
from typing import List

import torch
from torch.distributed import rpc

from hearthstone.asyncio import asyncio_utils
from hearthstone.ladder.ladder import Contestant
from hearthstone.simulator.host.async_host import AsyncHost
from hearthstone.simulator.replay.annotators.final_board_annotator import FinalBoardAnnotator
from hearthstone.simulator.replay.annotators.ranking_annotator import RankingAnnotator
from hearthstone.simulator.replay.replay import Replay
from hearthstone.training.pytorch.tensorboard_altair import TensorboardAltairAnnotator


class SimulationWorker:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        torch.set_num_threads(1)

    async def play_game(self, learning_bot_contestant, other_contestants, game_size):
        round_contestants = [learning_bot_contestant] + random.sample(other_contestants,
                                                                      k=game_size - 1)
        with torch.no_grad():
            host = AsyncHost(
                {contestant.name: contestant.agent_generator() for contestant in round_contestants},
                [RankingAnnotator(),
                 FinalBoardAnnotator(),
                 TensorboardAltairAnnotator([learning_bot_contestant.name])]
            )
            await host.async_play_game()
        return host.get_replay()

    def play_interleaved_games(self,
                               num_games: int,
                               learning_bot_contestant: Contestant,
                               other_contestants: List[Contestant],
                               game_size: int) -> List[Replay]:
        start = time.time()

        async def run_games():
            return await asyncio.gather(
                *[asyncio.create_task(self.play_game(learning_bot_contestant, other_contestants, game_size)) for _ in range(num_games)])

        replays = asyncio_utils.get_or_create_event_loop().run_until_complete(run_games())
        print("replays", replays)
        print(f"Worker played {num_games} game(s). Time taken: {time.time() - start} seconds.")
        return replays
