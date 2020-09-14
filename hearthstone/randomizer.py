import random
import typing
from typing import List, Tuple, Type, Optional

from hearthstone.monster_types import MONSTER_TYPES

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

    def select_friendly_minion(self, friendly_minions: List['MonsterCard']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_enemy_minion(self, enemy_minions: List['Card']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_discover_card(self, discoverables: List['Card']) -> 'Card':
        raise NotImplementedError()

    def select_from_store(self, store: List['Card']) -> 'Card':
        raise NotImplementedError()

    def select_gain_card(self, cards: List['MonsterCard']) -> 'Card':
        raise NotImplementedError()

    def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
        raise NotImplementedError()

    def select_summon_minion(self, cards: List['Type']) -> 'Type':
        raise NotImplementedError()

    def select_add_to_store(self, cards: List['Card']) -> 'Card':
        raise NotImplementedError()

    def select_monster_type(self, monster_types: List['MONSTER_TYPES'], round_number: int) -> 'MONSTER_TYPES':
        raise NotImplementedError()

    def select_random_minion(self, cards: List['Card'], round_number: int) -> 'Card':
        raise NotImplementedError()

    def select_adaptation(self, adaptations: List['Type']) -> 'Type':
        raise NotImplementedError()


class DefaultRandomizer(Randomizer):
    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.random()
        self.rand = random.Random(seed)

    def select_draw_card(self, cards: List['Card'], player_name: str, round_number: int) -> 'Card':
        return self.rand.choice(cards)

    def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
        random.shuffle(players)
        number_of_battles = len(players) // 2
        return list(zip(players[:number_of_battles], players[number_of_battles:]))

    def select_attack_target(self, defenders: List['Card']) -> 'Card':
        return self.rand.choice(defenders)

    def select_friendly_minion(self, friendly_minions: List['MonsterCard']) -> 'MonsterCard':
        return self.rand.choice(friendly_minions)

    def select_enemy_minion(self, enemy_minions: List['Card']) -> 'Card':
        return self.rand.choice(enemy_minions)

    def select_discover_card(self, discoverables: List['Card']) -> 'Card':
        return self.rand.choice(discoverables)

    def select_from_store(self, store: List['Card']) -> 'Card':
        return self.rand.choice(store)

    def select_gain_card(self, cards: List['Card']) -> 'Card':
        return self.rand.choice(cards)

    def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
        return self.rand.choice(hero_pool)

    def select_summon_minion(self, cards: List['Type']) -> 'Type':
        return self.rand.choice(cards)

    def select_add_to_store(self, cards: List['Card']) -> 'Card':
        return self.rand.choice(cards)

    def select_monster_type(self, monster_types: List['MONSTER_TYPES'], round_number: int) -> 'MONSTER_TYPES':
        return self.rand.choice(monster_types)

    def select_random_minion(self, cards: List['Card'], round_number: int) -> 'Card':
        return self.rand.choice(cards)

    def select_adaptation(self, adaptations: List['Type']) -> 'Type':
        return self.rand.choice(adaptations)
