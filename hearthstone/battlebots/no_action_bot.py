import typing
from typing import List

from hearthstone.simulator.agent import Agent, StandardAction, EndPhaseAction, DiscoverChoiceAction, \
    RearrangeCardsAction

if typing.TYPE_CHECKING:

    from hearthstone.simulator.core.player import Player


class NoActionBot(Agent):
    authors = ["Brian Kelly"]

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        return RearrangeCardsAction([])

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        return EndPhaseAction(True)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        return DiscoverChoiceAction(0)
