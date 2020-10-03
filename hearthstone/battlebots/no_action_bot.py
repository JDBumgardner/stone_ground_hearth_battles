import typing
from typing import List

from hearthstone.simulator.agent import Agent, Action, EndPhaseAction
if typing.TYPE_CHECKING:

    from hearthstone.simulator.core.player import Player


class NoActionBot(Agent):
    authors = ["Brian Kelly"]
    async def rearrange_cards(self, player: 'Player') -> List['MonsterCard']:
        return []

    async def buy_phase_action(self, player: 'Player') -> Action:
        return EndPhaseAction(True)

    async def discover_choice_action(self, player: 'Player') -> 'MonsterCard':
        return player.discover_queue[0][0]
