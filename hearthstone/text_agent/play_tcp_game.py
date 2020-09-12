import time
from typing import Dict

import trio

from hearthstone.agent import Agent
from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots.priority_functions import priority_saurolisk_bot, racist_priority_bot, \
    priority_adaptive_tripler_bot, battlerattler_priority_bot, priority_pogo_hopper_bot, priority_saurolisk_buff_bot
from hearthstone.host import RoundRobinHost, AsyncHost
from hearthstone.text_agent.stdio import StdIOTransport
from hearthstone.text_agent.tcp import TcpTransport
from hearthstone.text_agent.text_agent import TextAgent
from hearthstone.monster_types import MONSTER_TYPES
import logging


async def open_client_streams(max_players: int) -> Dict[str, Agent]:
    result = {}

    async with trio.open_nursery() as nursery:
        listeners = await trio.open_tcp_listeners(9998, host="0.0.0.0")
        while len(result) < max_players:
            stream = await listeners[0].accept()
            transport = TcpTransport(stream)
            await transport.send('Please enter your name: ')
            name = await transport.receive_line()
            name = name.strip()
            print(f"'{name}' has entered the game!")
            result[name] = TextAgent(transport)

    return result


def main():
    logging.basicConfig(level=logging.DEBUG)
    agents = {"battlerattler_priority_bot": battlerattler_priority_bot(1, EarlyGameBot),
              "priority_saurolisk_buff_bot": priority_saurolisk_buff_bot(2, EarlyGameBot),
              }
    agents.update(trio.run(open_client_streams, 8-len(agents)))
    host = AsyncHost(agents)
    host.play_game()


if __name__ == '__main__':
    main()
