import unittest

import logging

from hearthstone.simulator.core import player


class BattleGroundsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig(level=logging.DEBUG)
        player.TEST_MODE = True