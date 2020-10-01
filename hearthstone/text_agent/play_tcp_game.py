import asyncio
import logging
import socket
from typing import Dict

from hearthstone.agent import Agent
from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.host import AsyncHost
from hearthstone.text_agent.tcp import TcpTransport
from hearthstone.text_agent.text_agent import TextAgent


async def open_client_streams(max_players: int) -> Dict[str, Agent]:
    result = {}

    server: asyncio.AbstractServer

    async def serve(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter):
        nonlocal server
        transport = TcpTransport(client_reader, client_writer)
        await transport.send('Please enter your name: ')
        name = await transport.receive_line()
        name = name.strip()
        print(f"'{name}' has entered the game!")
        result[name] = TextAgent(transport)
        if len(result) >= max_players:
            server.close()

    server = await asyncio.start_server(serve, '0.0.0.0', 9998)
    await server.serve_forever()
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
