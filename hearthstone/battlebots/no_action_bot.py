import typing

from hearthstone.simulator.agent.actions import StandardAction, EndPhaseAction, DiscoverChoiceAction, \
    RearrangeCardsAction, FreezeDecision
from hearthstone.simulator.agent.agent import Agent

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.player import Player
from hearthstone.simulator.core.player import DiscoverIndex


class NoActionBot(Agent):
    authors = ["Brian Kelly"]

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        return RearrangeCardsAction([])

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        return EndPhaseAction(FreezeDecision.NO_FREEZE)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        return DiscoverChoiceAction(DiscoverIndex(0))
