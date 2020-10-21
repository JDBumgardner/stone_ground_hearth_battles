import itertools

from typing import Optional

from hearthstone.simulator.core.card_pool import *
from hearthstone.simulator.core.cards import MonsterCard, PrintingPress
from hearthstone.simulator.core.player import Player
from hearthstone.simulator.core.monster_types import MONSTER_TYPES


class CardHeuristic:
    cards_to_add: List[MonsterCard] = []

    def __init__(self, cards_to_add: Optional[List['MonsterCard']] = None):
        if cards_to_add:
            self.cards_to_add = cards_to_add

    def get_modification(self, player: 'Player', card: 'MonsterCard'):
        if type(card) in self.cards_to_add:
            return self.modification_value(player, card)
        return 0

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        pass


class SameTypeAdvantage(CardHeuristic):
    cards_to_add = [card_type for card_type in PrintingPress.all_types() if card_type.check_type(MONSTER_TYPES.BEAST)]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return len([player_card for player_card in itertools.chain(player.in_play, player.hand) if player_card.check_type(card.monster_type)])


class MamasLove(CardHeuristic):
    cards_to_add = [card_type for card_type in PrintingPress.all_types() if card_type.check_type(MONSTER_TYPES.BEAST)]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        if type(card) == MamaBear:
            return 20
        return 10 * len([player_card for player_card in itertools.chain(player.in_play, player.hand) if type(player_card) == MamaBear])


class DragonPayoffs(CardHeuristic):
    cards_to_add = [NadinaTheRed, RazorgoreTheUntamed]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        return 50 if len([player_card for player_card in itertools.chain(player.in_play, player.hand) if player_card.check_type(MONSTER_TYPES.DRAGON)]) >= 3 else 0


class MonstrousMacawPower(CardHeuristic):
    cards_to_add = [MonstrousMacaw]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        deathrattler_tiers = [player_card.tier for player_card in itertools.chain(player.in_play, player.hand) if player_card.deathrattles]
        if deathrattler_tiers:
            return max(deathrattler_tiers)
        return 0


class Scavengers(CardHeuristic):
    cards_to_add = [ScavengingHyena]

    def modification_value(self, player:'Player', card: 'MonsterCard'):
        if player.tavern_tier < 3:
            return len([player_card for player_card in itertools.chain(player.in_play, player.hand) if player_card.check_type(MONSTER_TYPES.BEAST)])