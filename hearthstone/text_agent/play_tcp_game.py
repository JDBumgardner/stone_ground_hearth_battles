import asyncio
import logging
import sys
from typing import Dict, Tuple

from hearthstone.simulator.agent import Agent
from hearthstone.simulator.host import CyborgArena
from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.text_agent.tcp import StoneProtocol, GameServer
from hearthstone.text_agent.text_agent import TextAgent
from hearthstone.simulator.core import hero_pool


async def open_client_streams(max_players: int) -> Dict[str, Agent]:
    loop = asyncio.get_event_loop()
    kill_event = asyncio.Event()
    game_server = GameServer(max_players, kill_event)

    async def serve():
        server = await loop.create_server(game_server.handle_connection, '0.0.0.0', 9998)
        async with server:
            await server.serve_forever()

    loop.create_task(serve())
    await game_server.wait_for_ready()
    result = {}
    for name, protocol in game_server.protocols.items():
        result[name] = TextAgent(protocol)
    return result


    # server: asyncio.AbstractServer
    #
    # async def serve(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter):
    #     if event.is_set():
    #         return
    #     nonlocal server
    #     transport = TcpTransport(client_reader, client_writer)
    #     await transport.send('Please enter your name: ')
    #     name = await transport.receive_line()
    #     name = name.strip()
    #     print(f"'{name}' has entered the game!")
    #     result[name] = TextAgent(transport)
    #     if len(result) >= max_players:
    #         print("at max players")
    #         event.set()
    #
    # server = await asyncio.start_server(serve, '0.0.0.0', 9998)
    # asyncio.get_event_loop().create_task(server.serve_forever())
    # await event.wait()
    # print("done waiting")
    # return result

def main():
    argv = sys.argv
    MAX_PLAYERS = 8
    if len(argv) > 1:
        MAX_PLAYERS = int(argv[1])

    logging.basicConfig(level=logging.DEBUG)
    agents = {
        "battlerattler_priority_bot": PriorityFunctions.battlerattler_priority_bot(1, EarlyGameBot),
        "priority_saurolisk_buff_bot": PriorityFunctions.priority_saurolisk_buff_bot(2, EarlyGameBot),
        "battlerattler_priority_bot2": PriorityFunctions.battlerattler_priority_bot(3, EarlyGameBot),
        "battlerattler_priority_bot3": PriorityFunctions.battlerattler_priority_bot(4, EarlyGameBot),
    }
    agents.update(asyncio.get_event_loop().run_until_complete(open_client_streams(MAX_PLAYERS-len(agents))))
    host = CyborgArena(agents)
    host.play_game()


if __name__ == '__main__':
    main()
