import random
import typing
from typing import List

from hearthstone.simulator.agent import Agent, StandardAction, generate_valid_actions, BuyAction, EndPhaseAction, \
    SummonAction, \
    TavernUpgradeAction, DiscoverChoiceAction, RearrangeCardsAction, HeroDiscoverAction

if typing.TYPE_CHECKING:

    from hearthstone.simulator.core.player import Player


class SupremacyBot(Agent):
    authors = ["Jeremy Salwen"]

    def __init__(self, monster_type: str, upgrade: bool, seed: int):
        self.local_random = random.Random(seed)
        self.monster_type = monster_type
        self.upgrade = upgrade

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        all_actions = list(generate_valid_actions(player))

        if self.upgrade:
            upgrade_actions = [action for action in all_actions if type(action) is TavernUpgradeAction]
            if upgrade_actions:
                return upgrade_actions[0]

        summon_actions = [action for action in all_actions if type(action) is SummonAction]
        if summon_actions:
            return summon_actions[0]

        buy_actions = [action for action in all_actions if type(action) is BuyAction and player.store[action.index].monster_type == self.monster_type]
        buy_actions = sorted(buy_actions, key=lambda buy_action: player.store[buy_action.index].tier, reverse=True)
        if buy_actions:
            return buy_actions[0]

        return EndPhaseAction(False)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        discover_cards = player.discover_queue[0]
        discover_cards = sorted(discover_cards, key=lambda card: card.tier, reverse=True)
        return DiscoverChoiceAction(player.discover_queue[0].index(discover_cards[0]))

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        return HeroDiscoverAction(self.local_random.choice(range(len(player.hero.discover_choices))))
