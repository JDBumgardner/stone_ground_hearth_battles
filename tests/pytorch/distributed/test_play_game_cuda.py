import logging
import unittest

import torch

from hearthstone.ladder.ladder import Contestant, ContestantAgentGenerator
from hearthstone.testing.battlegrounds_test_case import BattleGroundsTestCase
from hearthstone.training.pytorch.agents.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.encoding.default_encoder import DefaultEncoder
from hearthstone.training.pytorch.networks.transformer_net import HearthstoneTransformerNet
from hearthstone.training.pytorch.policy_gradient import easiest_contestants
from hearthstone.training.pytorch.worker.distributed.worker_pool import DistributedWorkerPool
from hearthstone.training.pytorch.worker.postprocessing import ReplaySink



class PytorchDistributedTests(BattleGroundsTestCase):
    def test_play_game_cuda(self):
        device = torch.device('cuda')
        p = DistributedWorkerPool(6,
                                  10,
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
