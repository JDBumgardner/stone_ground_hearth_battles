from hearthstone.host import RoundRobinHost
from hearthstone.user_agent import UserAgent
from hearthstone.card_pool import *


def main():
    host = RoundRobinHost({"dante_kong": UserAgent(), "hacker_on_steroids": UserAgent()})
    host.play_game()


if __name__ == '__main__':
    main()
