import sys
import unittest

import torch
from torch import multiprocessing

from hearthstone.ladder.ladder import Contestant, ContestantAgentGenerator
from hearthstone.training.pytorch.agents.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.encoding.default_encoder import DefaultEncoder
from hearthstone.training.pytorch.networks.transformer_net import HearthstoneTransformerNet
from hearthstone.training.pytorch.policy_gradient import easiest_contestants
from hearthstone.training.pytorch.worker.distributed.worker_pool import DistributedWorkerPool
from hearthstone.training.pytorch.worker.postprocessing import ReplaySink


def test_fn(rank):
    print(f"hello world {rank}")

class PytorchDistributedTests(unittest.TestCase):
    # def test_multiprocessing_forkserver(self):
    #     process_context = multiprocessing.start_processes(
    #         test_fn,
    #         nprocs=4,
    #         start_method="forkserver",
    #         join=False
    #     )
    #     process_context.join()
    #
    # def test_create_worker_pool(self):
    #     p = DistributedWorkerPool(5,
    #                               1,
    #                               True,
    #                               1024,
    #                               ReplaySink(),
    #                               torch.device('cpu'),
    #                               )
    #     p.shutdown()
    #
    # def test_play_game(self):
    #     p = DistributedWorkerPool(5,
    #                               1,
    #                               True,
    #                               1024,
    #                               ReplaySink(),
    #                               torch.device('cpu'))
    #     encoder = DefaultEncoder()
    #     learning_bot_contestant = Contestant(
    #         "LearningBot",
    #         ContestantAgentGenerator(PytorchBot,
    #                                  net=HearthstoneTransformerNet(encoder),
    #                                  encoder=encoder,
    #                                  annotate=True,
    #                                  device=torch.device('cpu'))
    #     )
    #     contestants = easiest_contestants()
    #     p.play_games(learning_bot_contestant=learning_bot_contestant, other_contestants=contestants, game_size=8)
    #     p.shutdown()


    def test_play_game_cuda(self):
        device = torch.device('cuda')
        p = DistributedWorkerPool(6,
                                  1024,
                                  True,
                                  1024,
                                  ReplaySink(),
                                  device)
        encoder = DefaultEncoder()
        learning_bot_contestant = Contestant(
            "LearningBot",
            ContestantAgentGenerator(PytorchBot,
                                     net=HearthstoneTransformerNet(encoder),
                                     encoder=encoder,
                                     annotate=True,
                                     device=torch.device('cpu'))
        )
        contestants = easiest_contestants()
        p.play_games(learning_bot_contestant=learning_bot_contestant, other_contestants=contestants, game_size=8)
        p.shutdown()


if __name__ == '__main__':
    unittest.main()
