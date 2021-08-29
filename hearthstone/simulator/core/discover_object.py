import enum
import itertools
import typing
from typing import Union, List, Callable

from hearthstone.simulator.core.cards import MonsterCard
from hearthstone.simulator.core.hero import Hero
from hearthstone.simulator.core.secrets import Secret
from hearthstone.simulator.core.spell import Spell

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.player import DiscoverIndex, Player

Discoverable = Union[MonsterCard, Hero, Spell, Secret]


class DiscoverType(enum.Enum):
    CARD = 0
    HERO = 1
    SECRET = 2
    SPELL = 3


class DiscoverObject:
    def __init__(self, items: List[Discoverable], discover_function: Callable[[Discoverable], None], dissolve_leftovers: bool, discover_type: DiscoverType):
        self.items = items
        self.discover_function = discover_function
        self.dissolve_leftovers = dissolve_leftovers
        self.discover_type = discover_type

    def select_item(self, index: 'DiscoverIndex', player: 'Player'):
        selected = self.items.pop(index)
        if isinstance(selected, MonsterCard):
            selected.token = False  # for Bigglesworth (there is no other scenario where a token will be a discover option)

        self.discover_function(selected)

        if self.dissolve_leftovers:
            assert all(isinstance(card, MonsterCard) for card in self.items)
            player.tavern.deck.return_cards(itertools.chain.from_iterable([card.dissolve() for card in self.items]))
