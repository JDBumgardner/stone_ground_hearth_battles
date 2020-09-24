import itertools

from typing import List, Callable, Dict, Optional

from hearthstone.card_pool import *
from hearthstone.cards import MonsterCard, PrintingPress
from hearthstone.player import Player
from hearthstone.monster_types import MONSTER_TYPES



class CardHeuristic:
    cards_to_add: List[MonsterCard] = []

    def __init__(self, cards_to_add: Optional[List['MonsterCard']] = None):
        if cards_to_add:
            self.cards_to_add = cards_to_add

    def get_modification(self, player: 'Player', card: 'MonsterCard'):
        if card in self.cards_to_add:
            return self.modification_value(player, card)
        return 0

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        pass


class SameTypeAdvantage(CardHeuristic):
    cards_to_add = [card_type for card_type in PrintingPress.all_types() if card_type.check_type(MONSTER_TYPES.BEAST)]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return len(player_card for player_card in itertools.chain(player.in_play, player.hand) if player_card.check_type(card.monster_type))


class MamasLove(CardHeuristic):
    cards_to_add = [card_type for card_type in PrintingPress.all_types() if card_type.check_type(MONSTER_TYPES.BEAST)]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return 10 * len(player_card for player_card in itertools.chain(player.in_play, player.hand) if type(player_card) == MamaBear)


class HoppingMad(CardHeuristic):
    cards_to_add = [PogoHopper]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return 5 * len(player_card for player_card in itertools.chain(player.in_play, player.hand) if type(player_card) == PogoHopper)


class DragonPayoffs(CardHeuristic):
    cards_to_add = [NadinaTheRed, RazorgoreTheUntamed]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return 50 if len(player_card for player_card in itertools.chain(player.in_play, player.hand) if player_card.check_type(MONSTER_TYPES.DRAGON)) >= 3 else 0

class MonstrousMacawPower(CardHeuristic):
    cards_to_add = [MonstrousMacaw]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return max(player_card.tier for player_card in itertools.chain(player.in_play, player.hand) if player_card.deathrattles)
