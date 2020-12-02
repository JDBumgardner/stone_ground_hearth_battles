import concurrent.futures
import multiprocessing.pool
import random
import time
from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter

from hearthstone.ladder.ladder import Contestant, update_ratings, print_standings
from hearthstone.simulator.host.round_robin_host import RoundRobinHost
from hearthstone.simulator.replay.annotators.final_board_annotator import FinalBoardAnnotator
from hearthstone.simulator.replay.annotators.ranking_annotator import RankingAnnotator
from hearthstone.training.pytorch import tensorboard_altair
from hearthstone.training.pytorch.encoding import shared_tensor_pool_encoder
from hearthstone.training.pytorch.encoding.shared_tensor_pool_encoder import SharedTensorPoolEncoder
from hearthstone.training.pytorch.gae import GAEAnnotator
from hearthstone.training.pytorch.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.replay_buffer import EpochBuffer
from hearthstone.training.pytorch.surveillance import GlobalStepContext
from hearthstone.training.pytorch.tensorboard_altair import TensorboardAltairAnnotator


def play_game(learning_bot_contestant: Contestant,
              other_contestants: List[Contestant],
              game_size: int,
              annotator: GAEAnnotator):
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
        annotator.annotate(replay)
        return replay


class WorkerPool:
    def __init__(self, num_workers,
                 epoch_buffer: EpochBuffer,
                 annotator: GAEAnnotator,
                 encoder: SharedTensorPoolEncoder,
                 tensorboard: SummaryWriter,
                 global_step_context: GlobalStepContext,
                 use_processes: bool
                 ):
        self.num_workers = num_workers
        self.use_processes = use_processes
        if use_processes:
            def setup_worker_process(q):
                """Copies the global queue from the parent process to the child processes, overwriting the child's"""
                torch.set_num_threads(1)  # This is really important, otherwise OpenMP messes things up.
                shared_tensor_pool_encoder.gloabal_tensor_queue = q

            self.pool = torch.multiprocessing.Pool(initializer=setup_worker_process(),
                                              initargs=(shared_tensor_pool_encoder.global_tensor_queue,),
                                              processes=num_workers)
        else:
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.epoch_buffer = epoch_buffer
        self.annotator = annotator
        self.encoder = encoder
        self.tensorboard = tensorboard
        self.global_step_context = global_step_context

    def _submit_task(self, fn, args):
        if self.use_processes:
            return self.pool.apply_async(fn, args)
        else:
            return self.pool.submit(fn, *args)

    def _get_task_result(self, promise):
        if self.use_processes:
            return promise.get()
        else:
            return promise.result()

    def play_games(self, learning_bot_contestant: Contestant, other_contestants: List[Contestant], game_size: int):
        num_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        all_contestants = [learning_bot_contestant] + other_contestants
        if self.use_processes:
            for contestant in all_contestants:
                if contestant.agent_generator.function == PytorchBot:
                    contestant.agent_generator().net.share_memory()
                    print(contestant)
        with torch.no_grad():
            awaitables = [
                self._submit_task(play_game, (learning_bot_contestant, other_contestants, game_size, self.annotator))
                for _ in
                range(self.num_workers)]
            for promise in awaitables:
                replay = self._get_task_result(promise)
                tensorboard_altair.plot_replay(replay, learning_bot_contestant.name, self.tensorboard,
                                               self.global_step_context)
                self._update_ratings(learning_bot_contestant, all_contestants, replay)
                self.epoch_buffer.add_replay(replay)
        torch.set_num_threads(num_torch_threads)

    @staticmethod
    def _update_ratings(learning_bot_contestant, all_contestants, replay):
        winner_names = replay.observer_annotations["RankingAnnotator"]
        final_boards = replay.observer_annotations["FinalBoardAnnotator"]
        print("---------------------------------------------------------------")
        print(winner_names)
        print("["+", ".join(final_boards[learning_bot_contestant.name])+"]")
        ranked_contestants = sorted([c for c in all_contestants if c.name in winner_names],
                                    key=lambda c: winner_names.index(c.name))
        update_ratings(ranked_contestants)
        print_standings(all_contestants)
        for contestant in ranked_contestants:
            contestant.games_played += 1
