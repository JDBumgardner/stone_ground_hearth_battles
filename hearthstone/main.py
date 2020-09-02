from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.battlebots.priority_functions import priority_saurolisk_bot, racist_priority_bot, \
    priority_adaptive_tripler_bot, battlerattler_priority_bot, priority_pogo_hopper_bot, priority_saurolisk_buff_bot
from hearthstone.host import AsyncHost
from hearthstone.user_agent import UserAgent
from hearthstone.monster_types import MONSTER_TYPES
import logging


def main():
    logging.basicConfig(level=logging.DEBUG)
    host = AsyncHost({"dante_kong": UserAgent(),
                      # "david_stolfo": UserAgent(),
                      "no_action_bot": NoActionBot(),
                      "battlerattler_priority_bot": battlerattler_priority_bot(1, EarlyGameBot),
                      "priority_saurolisk_buff_bot": priority_saurolisk_buff_bot(2, EarlyGameBot),
                      "racist_priority_bot_mech": racist_priority_bot(3, EarlyGameBot, MONSTER_TYPES.MECH),
                      "racist_priority_bot_murloc": racist_priority_bot(4, EarlyGameBot, MONSTER_TYPES.MURLOC),
                      "priority_adaptive_tripler_bot": priority_adaptive_tripler_bot(5, EarlyGameBot),
                      "priority_pogo_hopper_bot": PriorityFunctions.priority_pogo_hopper_bot(7, PriorityBot),
                      })
    host.play_game()


if __name__ == '__main__':
    main()
