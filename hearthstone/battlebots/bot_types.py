import random
import typing
from typing import List, Callable

from hearthstone.simulator.agent import Agent, Action
if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.cards import MonsterCard
    from hearthstone.simulator.core.player import Player


class PriorityFunctionBot(Agent):
    def __init__(self, authors: List[str], priority: Callable[['Player', 'MonsterCard'], float], seed: int):
        if not authors:
            authors = ["JB", "AS", "ES", "JS", "DVP"]
        self.authors = authors
        self.priority = priority
        self.local_random = random.Random(seed)

    async def discover_choice_action(self, player: 'Player') -> 'MonsterCard':
        discover_cards = player.discover_queue[0]
        discover_cards = sorted(discover_cards, key=lambda card: self.priority(player, card), reverse=True)
        return discover_cards[0]

    async def rearrange_cards(self, player: 'Player') -> List['MonsterCard']:
        card_list = player.in_play.copy()
        self.local_random.shuffle(card_list)
        return card_list

    async def buy_phase_action(self, player: 'Player') -> Action:
        pass
