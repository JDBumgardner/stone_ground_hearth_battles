import asyncio
import logging
import sys
from typing import Dict, Tuple

from hearthstone.simulator.agent import Agent
from hearthstone.simulator.host.cyborg_host import CyborgArena
from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.text_agent.tcp import GameServer
from hearthstone.text_agent.text_agent import TextAgent
from hearthstone.simulator.core import hero_pool




async def open_client_streams(max_players: int) -> Dict[str, Agent]:
    kill_event = asyncio.Event()
    game_server = GameServer(max_players, kill_event)
    game_server.serve_forever()
    await game_server.wait_for_ready()
    return {
        name: TextAgent(protocol)
        for name, protocol in game_server.protocols.items()
    }


def main():
    argv = sys.argv
    MAX_PLAYERS = 8
    if len(argv) > 1:
        MAX_PLAYERS = int(argv[1])

    logging.basicConfig(level=logging.DEBUG)
    agents = {
        "battlerattler_priority_bot": PriorityFunctions.battlerattler_priority_bot(1, EarlyGameBot),
        "battlerattler_priority_bot2": PriorityFunctions.battlerattler_priority_bot(3, EarlyGameBot),
        "battlerattler_priority_bot3": PriorityFunctions.battlerattler_priority_bot(4, EarlyGameBot),
        "battlerattler_priority_bot4": PriorityFunctions.battlerattler_priority_bot(5, EarlyGameBot),
        "battlerattler_priority_bot5": PriorityFunctions.battlerattler_priority_bot(6, EarlyGameBot),
        "battlerattler_priority_bot6": PriorityFunctions.battlerattler_priority_bot(7, EarlyGameBot),
        "battlerattler_priority_bot7": PriorityFunctions.battlerattler_priority_bot(8, EarlyGameBot),
    }
    agents.update(asyncio.get_event_loop().run_until_complete(open_client_streams(MAX_PLAYERS-len(agents))))
    host = CyborgArena(agents)
    host.play_game()


if __name__ == '__main__':
    main()
