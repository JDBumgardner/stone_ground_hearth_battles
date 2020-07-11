import random
import typing
from typing import List, Tuple

if typing.TYPE_CHECKING:
    from hearthstone.cards import MonsterCard, Card
    from hearthstone.hero import Hero
    from hearthstone.player import Player


class Randomizer:
    def select_draw_card(self, cards: List['Card'], player_name: str, round_number: int) -> 'Card':
        raise NotImplementedError()

    def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
        raise NotImplementedError()

    def select_attack_target(self, defenders: List['Card']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_friendly_minion(self, friendly_minions: List['Card']) -> 'Card':
        raise NotImplementedError()

    def select_enemy_minion(self, enemy_minions: List['Card']) -> 'Card':
        raise NotImplementedError()

    def select_discover_card(self, discoverables: List['Card']) -> 'Card':
        raise NotImplementedError()

    def select_from_store(self, store: List['Card']) -> 'Card':
        raise NotImplementedError()

    def select_gain_card(self, cards: List['Card']) -> 'Card':
        raise NotImplementedError()

    def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
        raise NotImplementedError()


class DefaultRandomizer(Randomizer):
    def select_draw_card(self, cards: List['Card'], player_name: str, round_number: int) -> 'Card':
        return random.choice(cards)

    def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
        random.shuffle(players)
        number_of_battles = len(players) // 2
        return list(zip(players[:number_of_battles], players[number_of_battles:]))

    def select_attack_target(self, defenders: List['Card']) -> 'Card':
        return random.choice(defenders)

    def select_friendly_minion(self, friendly_minions: List['Card']) -> 'Card':
        return random.choice(friendly_minions)

    def select_enemy_minion(self, enemy_minions: List['Card']) -> 'Card':
        return random.choice(enemy_minions)

    def select_discover_card(self, discoverables: List['Card']) -> 'Card':
        return random.choice(discoverables)

    def select_from_store(self, store: List['Card']) -> 'Card':
        return random.choice(store)

    def select_gain_card(self, cards: List['Card']) -> 'Card':
        return random.choice(cards)

    def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
        return random.choice(hero_pool)