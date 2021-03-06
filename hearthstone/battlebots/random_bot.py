import random
import typing
from typing import List

from hearthstone.simulator.agent import Agent, generate_valid_actions, StandardAction, DiscoverChoiceAction, \
    RearrangeCardsAction, HeroDiscoverAction

if typing.TYPE_CHECKING:

    from hearthstone.simulator.core.player import Player


class RandomBot(Agent):
    authors = ["Jeremy Salwen"]

    def __init__(self, seed: int):
        self.local_random = random.Random(seed)

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        all_actions = list(generate_valid_actions(player))
        return self.local_random.choice(all_actions)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        return DiscoverChoiceAction(self.local_random.choice(range(len(player.discover_queue[0]))))

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        return HeroDiscoverAction(self.local_random.choice(range(len(player.hero.discover_choices))))
