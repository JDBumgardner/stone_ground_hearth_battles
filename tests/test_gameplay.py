import unittest

from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.simulator.host import AsyncHost
from hearthstone.simulator.core.monster_types import MONSTER_TYPES


class GameplayTests(unittest.TestCase):
    def test_basic_bots(self):
        host = AsyncHost({
            "battlerattler_priority_bot": PriorityFunctions.battlerattler_priority_bot(1, EarlyGameBot),
            "priority_saurolisk_buff_bot": PriorityFunctions.priority_saurolisk_buff_bot(2, EarlyGameBot),
            "racist_priority_bot_mech": PriorityFunctions.racist_priority_bot(3, EarlyGameBot, MONSTER_TYPES.MECH),
            "racist_priority_bot_murloc": PriorityFunctions.racist_priority_bot(4, EarlyGameBot, MONSTER_TYPES.MURLOC),
            "priority_adaptive_tripler_bot": PriorityFunctions.priority_adaptive_tripler_bot(5, EarlyGameBot),
            "priority_pogo_hopper_bot": PriorityFunctions.priority_pogo_hopper_bot(7, PriorityBot),
        })
        host.play_game()


if __name__ == '__main__':
    unittest.main()
