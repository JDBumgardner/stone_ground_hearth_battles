from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.priority_bot import priority_saurolisk_bot, racist_priority_bot, \
    priority_adaptive_tripler_bot, battlerattler_priority_bot
from hearthstone.battlebots.priority_storage_bot import priority_st_ad_tr_bot
from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.host import RoundRobinHost
from hearthstone.user_agent import UserAgent
from hearthstone.card_pool import *
import logging



def main():
    logging.basicConfig(level=logging.DEBUG)

    host = RoundRobinHost({"random_action_bot":RandomBot(2),
                           "my_bot":priority_st_ad_tr_bot(1)
                           })
    host.play_game()


if __name__ == '__main__':
    main()
