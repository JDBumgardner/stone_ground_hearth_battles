import random
import typing
from typing import List

from hearthstone.agent import Agent, generate_valid_actions, Action
if typing.TYPE_CHECKING:

    from hearthstone.player import Player


class RandomBot(Agent):
    authors = ["Jeremy Salwen"]
    def __init__(self, seed: int):
        self.local_random = random.Random(seed)

    async def rearrange_cards(self, player: 'Player') -> List['MonsterCard']:
        card_list = player.in_play.copy()
        self.local_random.shuffle(card_list)
        return card_list

    async def buy_phase_action(self, player: 'Player') -> Action:
        all_actions = list(generate_valid_actions(player))
        return self.local_random.choice(all_actions)

    async def discover_choice_action(self, player: 'Player') -> 'MonsterCard':
        return self.local_random.choice(player.discover_queue[0])
