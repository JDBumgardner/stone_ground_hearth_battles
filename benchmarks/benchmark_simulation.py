import asyncio

from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.simulator.host.async_host import AsyncHost
import cProfile

async def main():
    hosts = [AsyncHost({f'RandomBot{i}': RandomBot(i+j) for i in range(8)})
             for j in range(25)
             ]
    tasks = [asyncio.create_task(host.async_play_game()) for host in hosts]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main())
