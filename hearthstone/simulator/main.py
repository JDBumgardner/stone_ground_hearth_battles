import logging

from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.host.async_host import AsyncHost
from hearthstone.text_agent.stdio import StdIOTransport
from hearthstone.text_agent.text_agent import TextAgent


def main():
    logging.basicConfig(level=logging.DEBUG)
    host = AsyncHost({"dante_kong": TextAgent(StdIOTransport()),
                      "david_stolfo": CheapoBot(1),
                      "battlerattler_priority_bot": PriorityFunctions.battlerattler_priority_bot(2, EarlyGameBot),
                      "priority_saurolisk_buff_bot": PriorityFunctions.priority_saurolisk_buff_bot(3, EarlyGameBot),
                      "racist_priority_bot_mech": PriorityFunctions.racist_priority_bot(4, EarlyGameBot,
                                                                                        MONSTER_TYPES.MECH),
                      "racist_priority_bot_murloc": PriorityFunctions.racist_priority_bot(5, EarlyGameBot,
                                                                                          MONSTER_TYPES.MURLOC),
                      "racist_priority_bot_elemental": PriorityFunctions.racist_priority_bot(6, EarlyGameBot,
                                                                                             MONSTER_TYPES.ELEMENTAL),
                      "priority_adaptive_tripler_bot": PriorityFunctions.priority_adaptive_tripler_bot(7, EarlyGameBot),
                      })
    host.play_game()


if __name__ == '__main__':
    main()
