import logging
from hearthstone.battlebots.priority_storage_bot import priority_st_ad_tr_bot
from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.host import RoundRobinHost


def main():
    logging.basicConfig(level=logging.DEBUG)

    host = RoundRobinHost({"random_action_bot":RandomBot(2),
                           "my_bot":priority_st_ad_tr_bot(1)
                           })
    host.play_game()


if __name__ == '__main__':
    main()
