import asyncio
import logging
import socket
from typing import Dict

from hearthstone.agent import Agent
from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.host import AsyncHost
from hearthstone.text_agent.text_agent import TextAgent


async def open_client_streams(max_players: int) -> Dict[str, Agent]:
    result = {}

    loop = asyncio.get_event_loop()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 9998))
    server.listen(8)
    server.setblocking(False)
    while len(result) < max_players:
        conn, _ = await loop.sock_accept(server)
        my_stream_reader = WHAT(conn)
        await transport.send('Please enter your name: ')
        name = await transport.receive_line()
        name = name.strip()
        print(f"'{name}' has entered the game!")
        result[name] = TextAgent(transport)

    return result


def main():
    logging.basicConfig(level=logging.DEBUG)
    agents = {
        "battlerattler_priority_bot": PriorityFunctions.battlerattler_priority_bot(1, EarlyGameBot),
        "priority_saurolisk_buff_bot": PriorityFunctions.priority_saurolisk_buff_bot(2, EarlyGameBot),
    }
    agents.update(asyncio.get_event_loop().run_until_complete(open_client_streams(8-len(agents))))
    host = AsyncHost(agents)
    host.play_game()


if __name__ == '__main__':
    main()
