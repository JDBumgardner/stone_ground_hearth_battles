import asyncio
import random
import time
from typing import List

import logging
import torch
from torch.distributed import rpc

from hearthstone.asyncio import asyncio_utils
from hearthstone.ladder.ladder import Contestant
from hearthstone.simulator.host.async_host import AsyncHost
from hearthstone.simulator.replay.annotators.final_board_annotator import FinalBoardAnnotator
from hearthstone.simulator.replay.annotators.ranking_annotator import RankingAnnotator
from hearthstone.simulator.replay.replay import Replay
from hearthstone.training.pytorch.agents.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.tensorboard_altair import TensorboardAltairAnnotator
from hearthstone.training.pytorch.worker.distributed.remote_net import RemoteNet, BatchedRemoteNet

logger = logging.getLogger(__name__)

import tracemalloc
tracemalloc.start(25)
snapshot1 = None

class SimulationWorker:
    def __init__(self, inference_worker):
        self.id = rpc.get_worker_info().id
        self.inference_worker = inference_worker
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
        global snapshot1
        start = time.time()

        print(f'all tasks {len(asyncio.all_tasks(asyncio_utils.get_or_create_event_loop()))}')
        for task in asyncio.all_tasks(asyncio_utils.get_or_create_event_loop()):
            asyncio_utils.show_coro(task)

        snapshot2 = tracemalloc.take_snapshot()
        if snapshot1:
            top_stats = snapshot2.compare_to(snapshot1, 'traceback')
            for entry in top_stats[:10]:
                print('\nEntry: {}'.format(entry))
                print('Traceback:')
                for line in entry.traceback:
                    print('  {}'.format(line))
        snapshot1 = snapshot2

        async def run_games():
            nets = {}
            for contestant in [learning_bot_contestant] + other_contestants:
                if contestant.agent_generator.function == PytorchBot:
                    if type(contestant.agent_generator.kwargs['net']) is RemoteNet:
                        if contestant.name not in nets:
                            nets[contestant.name] = BatchedRemoteNet(contestant.name, self.inference_worker)
                        contestant.agent_generator.kwargs['net'] = nets[contestant.name]
            for _, net in nets.items():
                await net.start_worker()
            tasks = [asyncio_utils.create_task(self.play_game(learning_bot_contestant, other_contestants, game_size), logger=logger) for _ in
                     range(num_games)]
            result = await asyncio.gather(
                *tasks)
            for _, net in nets.items():
                await net.stop_worker()
            return result

        replays = asyncio_utils.get_or_create_event_loop().run_until_complete(run_games())
        print(f"Worker played {num_games} game(s). Time taken: {time.time() - start} seconds.")
        return replays
