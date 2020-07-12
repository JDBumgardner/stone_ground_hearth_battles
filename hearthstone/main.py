from hearthstone.battlebots.priority_functions import priority_saurolisk_bot, racist_priority_bot, \
    priority_adaptive_tripler_bot, battlerattler_priority_bot
from hearthstone.host import RoundRobinHost
from hearthstone.user_agent import UserAgent
from hearthstone.monster_types import MURLOC, DEMON, MECH
import logging


def main():
    logging.basicConfig(level = logging.DEBUG)
    host = RoundRobinHost({"dante_kong": UserAgent(),
                           "hacker_on_steroids": UserAgent(),
                           "PrioritySauroliskBot": priority_saurolisk_bot(1),
                           "PriorityMurlocBot": racist_priority_bot(MURLOC, 1),
                           "PriorityDemonBot": racist_priority_bot(DEMON, 1),
                           "PriorityMechBot": racist_priority_bot(MECH, 1),
                           "PriorityAdaptiveTriplerBot": priority_adaptive_tripler_bot(1),
                           "BattleRattlerBot": battlerattler_priority_bot(1),
                           })
    host.play_game()


if __name__ == '__main__':
    main()
