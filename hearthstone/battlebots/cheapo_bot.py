import random
import typing

from hearthstone.simulator.agent.actions import StandardAction, generate_standard_actions, BuyAction, EndPhaseAction, \
    SummonAction, DiscoverChoiceAction, RearrangeCardsAction, FreezeDecision
from hearthstone.simulator.agent.agent import Agent
from hearthstone.simulator.core.discover_object import DiscoverType

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.player import Player, DiscoverIndex


class CheapoBot(Agent):
    authors = ["Brian Kelly"]

    def __init__(self, seed: int):
        self.local_random = random.Random(seed)

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        all_actions = list(generate_standard_actions(player))

        summon_actions = [action for action in all_actions if type(action) is SummonAction]
        if summon_actions:
            return summon_actions[0]

        buy_actions = [action for action in all_actions if type(action) is BuyAction]
        buy_actions = sorted(buy_actions, key=lambda buy_action: player.store[buy_action.index].tier)
        if buy_actions:
            return buy_actions[0]

        return EndPhaseAction(FreezeDecision.NO_FREEZE)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        discover_object = player.discover_queue[0]
        if discover_object.discover_type == DiscoverType.CARD:
            discover_object.items.sort(key=lambda card: card.tier)
        return DiscoverChoiceAction(DiscoverIndex(discover_object.items.index(discover_object.items[0])))
