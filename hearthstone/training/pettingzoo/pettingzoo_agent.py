import asyncio

from hearthstone.simulator.agent.actions import StandardAction, RearrangeCardsAction, DiscoverChoiceAction
from hearthstone.simulator.agent.agent import Agent


class AgentRequestQueue:
    """
    A class for passing data back and forth between the PettingZoo API and the PettingZooAgents.
    The `requests` queue contains tuples of (player_name, Future)
    """
    def __init__(self, maxsize: int = 8):
        self.requests: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def request_agent_action(self, player_name: str):
        future = asyncio.Future()
        self.requests.put_nowait((player_name, future))
        return await future




class PettingZooAgent(Agent):
    def __init__(self, queue: AgentRequestQueue):
        self.queue = queue

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        queu

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        pass

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        pass
