import typing
from typing import List

from hearthstone.agent import Agent, Action, EndPhaseAction
if typing.TYPE_CHECKING:
    from hearthstone.cards import Card
    from hearthstone.player import Player


class NoActionBot(Agent):
    authors = ["Brian Kelly"]
    async def rearrange_cards(self, player: 'Player') -> List['Card']:
        return []

    async def buy_phase_action(self, player: 'Player') -> Action:
        return EndPhaseAction(True)

    async def discover_choice_action(self, player: 'Player') -> 'Card':
        return player.discover_queue[0][0]
