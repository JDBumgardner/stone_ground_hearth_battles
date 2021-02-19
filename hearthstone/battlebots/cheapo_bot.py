import random
import typing
from typing import List

from hearthstone.simulator.agent import Agent, StandardAction, generate_valid_actions, BuyAction, EndPhaseAction, \
    SummonAction, DiscoverChoiceAction, RearrangeCardsAction, HeroDiscoverAction, FreezeDecision

if typing.TYPE_CHECKING:

    from hearthstone.simulator.core.player import Player


class CheapoBot(Agent): 
    authors = ["Brian Kelly"]

    def __init__(self, seed: int):
        self.local_random = random.Random(seed)

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        all_actions = list(generate_valid_actions(player))

        summon_actions = [action for action in all_actions if type(action) is SummonAction]
        if summon_actions:
            return summon_actions[0]

        buy_actions = [action for action in all_actions if type(action) is BuyAction]
        buy_actions = sorted(buy_actions, key=lambda buy_action: player.store[buy_action.index].tier)
        if buy_actions:
            return buy_actions[0]

        return EndPhaseAction(FreezeDecision.NO_FREEZE)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        discover_cards = player.discover_queue[0]
        discover_cards = sorted(discover_cards, key=lambda card: card.tier)
        return DiscoverChoiceAction(player.discover_queue[0].index(discover_cards[0]))

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        return HeroDiscoverAction(self.local_random.choice(range(len(player.hero.discover_choices))))
