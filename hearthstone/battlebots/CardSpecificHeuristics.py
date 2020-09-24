import itertools
from typing import List, Callable, Dict

from hearthstone.card_pool import MamaBear
from hearthstone.cards import MonsterCard, PrintingPress
from hearthstone.player import Player
from hearthstone.monster_types import MONSTER_TYPES



class CardHeuristic:
    cards_to_add: List[MonsterCard] = []

    def __init__(self, cards_to_add: 'MonsterCard' = None):
        if cards_to_add:
            self.cards_to_add = cards_to_add

    def get_modification(self, player: 'Player', card: 'MonsterCard'):
        if card in self.cards_to_add:
            return self.modification_value(player, card)
        return 0

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        pass


class BeastBump(CardHeuristic):
    cards_to_add = [card_type for card_type in PrintingPress.all_types() if card_type.check_type(MONSTER_TYPES.BEAST)]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return len([card for card in itertools.chain(*player.in_play, *player.hand) if card.check_type(MONSTER_TYPES.BEAST)])


class MamasLove(CardHeuristic):
    cards_to_add = [card_type for card_type in PrintingPress.all_types() if card_type.check_type(MONSTER_TYPES.BEAST)]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return 10 * len([card for card in itertools.chain(*player.in_play, *player.hand) if type(card) == MamaBear])