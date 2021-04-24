import unittest

import logging

from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.randomizer import DefaultRandomizer
from hearthstone.simulator.host.async_host import AsyncHost
from hearthstone.testing.battlegrounds_test_case import BattleGroundsTestCase


class GameplayTests(BattleGroundsTestCase):
    def test_basic_bots(self):
        host = AsyncHost({
            "battlerattler_priority_bot": PriorityFunctions.battlerattler_priority_bot(1, EarlyGameBot),
            "priority_saurolisk_buff_bot": PriorityFunctions.priority_saurolisk_buff_bot(2, EarlyGameBot),
            "racist_priority_bot_mech": PriorityFunctions.racist_priority_bot(3, EarlyGameBot, MONSTER_TYPES.MECH),
            "racist_priority_bot_murloc": PriorityFunctions.racist_priority_bot(4, EarlyGameBot, MONSTER_TYPES.MURLOC),
            "priority_adaptive_tripler_bot": PriorityFunctions.priority_adaptive_tripler_bot(5, EarlyGameBot),
            "priority_pack_leader_bot": PriorityFunctions.priority_pack_leader_bot(7, PriorityBot),
        }, randomizer=DefaultRandomizer(107))

        host.play_game()

    def test_replay_same_outcome(self):
        logging.basicConfig(level=logging.DEBUG)
        # TODO make replays work so this test passes.  This requires handling the ordering of players joining and
        # Choosing their heros.
        host = AsyncHost({
            "battlerattler_priority_bot": PriorityFunctions.battlerattler_priority_bot(1, EarlyGameBot),
            "priority_saurolisk_buff_bot": PriorityFunctions.priority_saurolisk_buff_bot(2, EarlyGameBot),
            "racist_priority_bot_mech": PriorityFunctions.racist_priority_bot(3, EarlyGameBot, MONSTER_TYPES.MECH),
            "racist_priority_bot_murloc": PriorityFunctions.racist_priority_bot(4, EarlyGameBot, MONSTER_TYPES.MURLOC),
            "priority_adaptive_tripler_bot": PriorityFunctions.priority_adaptive_tripler_bot(5, EarlyGameBot),
            "priority_pack_leader_bot": PriorityFunctions.priority_pack_leader_bot(7, PriorityBot),
        }, randomizer=DefaultRandomizer(11))
        host.play_game()
        replay = host.get_replay()
        replayed_tavern = replay.run_replay()
        self.assertListEqual([name for name, _ in host.tavern.losers], [name for name, _ in replayed_tavern.losers])


if __name__ == '__main__':
    unittest.main()
