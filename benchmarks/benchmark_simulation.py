import asyncio

from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.simulator.host.async_host import AsyncHost


async def main():
    hosts = [AsyncHost({f'RandomBot{i}': RandomBot(i+j) for i in range(8)})
             for j in range(25)
             ]
    tasks = [asyncio.create_task(host.async_play_game()) for host in hosts]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
