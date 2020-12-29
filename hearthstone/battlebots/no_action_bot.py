import typing
from typing import List

from hearthstone.simulator.agent import Agent, StandardAction, EndPhaseAction, DiscoverChoiceAction, \
    RearrangeCardsAction, HeroDiscoverAction

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.player import Player
from hearthstone.simulator.core.player import DiscoverIndex


class NoActionBot(Agent):
    authors = ["Brian Kelly"]

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        return RearrangeCardsAction([])

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        return EndPhaseAction(True)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        return DiscoverChoiceAction(DiscoverIndex(0))

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        return HeroDiscoverAction(DiscoverIndex(0))
