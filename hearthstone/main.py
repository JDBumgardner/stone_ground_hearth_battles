from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots.priority_functions import priority_saurolisk_bot, racist_priority_bot, \
    priority_adaptive_tripler_bot, battlerattler_priority_bot
from hearthstone.host import RoundRobinHost
from hearthstone.user_agent import UserAgent
from hearthstone.monster_types import MONSTER_TYPES
import logging


def main():
    logging.basicConfig(level = logging.DEBUG)
    host = RoundRobinHost({"dante_kong": UserAgent(),
                           "david_stolfo": UserAgent(),
                           "no_action_bot": NoActionBot(),
                           "battlerattler_priority_bot": battlerattler_priority_bot(2, PriorityBot),
                           "priority_saurolisk_bot": priority_saurolisk_bot(1, PriorityBot),
                           "racist_priority_bot_mech": racist_priority_bot(3, PriorityBot, MONSTER_TYPES.MECH),
                           "racist_priority_bot_murloc": racist_priority_bot(3, PriorityBot, MONSTER_TYPES.MURLOC),
                           "priority_adaptive_tripler_bot": priority_adaptive_tripler_bot(4, PriorityBot)
                           })
    host.play_game()


if __name__ == '__main__':
    main()
