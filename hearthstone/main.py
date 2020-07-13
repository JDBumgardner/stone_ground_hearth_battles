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
                           })
    host.play_game()


if __name__ == '__main__':
    main()
