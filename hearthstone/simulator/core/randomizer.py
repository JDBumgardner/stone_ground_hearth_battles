import random
import typing
from typing import List, Tuple, Type, Optional

from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.secrets import SECRETS

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.cards import MonsterCard
    from hearthstone.simulator.core.hero import Hero
    from hearthstone.simulator.core.player import Player


class Randomizer:
    def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
        raise NotImplementedError()

    def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
        raise NotImplementedError()

    def select_attack_target(self, defenders: List['MonsterCard']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_friendly_minion(self, friendly_minions: List['MonsterCard']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_enemy_minion(self, enemy_minions: List['MonsterCard']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_discover_card(self, discoverables: List['MonsterCard']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_from_store(self, store: List['MonsterCard']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_gain_card(self, cards: List['MonsterCard']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
        raise NotImplementedError()

    def select_summon_minion(self, cards: List['Type']) -> 'Type':
        raise NotImplementedError()

    def select_add_to_store(self, cards: List['MonsterCard']) -> 'MonsterCard':
        raise NotImplementedError()

    def select_monster_type(self, monster_types: List['MONSTER_TYPES'], round_number: int) -> 'MONSTER_TYPES':
        raise NotImplementedError()

    def select_random_minion(self, cards: List['Type'], round_number: int) -> 'Type':
        raise NotImplementedError()

    def select_adaptation(self, adaptations: List['Type']) -> 'Type':
        raise NotImplementedError()

    def select_random_number(self, lo: int, hi: int) -> int:
        raise NotImplementedError()

    def select_secret(self, secrets: List['SECRETS']) -> 'SECRETS':
        raise NotImplementedError()

    def select_combat_matchup(self, pairings: List[Tuple['Player', 'Player']]) -> Tuple['Player', 'Player']:
        raise NotImplementedError()


class DefaultRandomizer(Randomizer):
    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.random()
        self.seed = seed
        self.rand = random.Random(seed)

    def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
        return self.rand.choice(cards)

    def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
        random.shuffle(players)
        number_of_battles = len(players) // 2
        return list(zip(players[:number_of_battles], players[number_of_battles:]))

    def select_attack_target(self, defenders: List['MonsterCard']) -> 'MonsterCard':
        return self.rand.choice(defenders)

    def select_friendly_minion(self, friendly_minions: List['MonsterCard']) -> 'MonsterCard':
        return self.rand.choice(friendly_minions)

    def select_enemy_minion(self, enemy_minions: List['MonsterCard']) -> 'MonsterCard':
        return self.rand.choice(enemy_minions)

    def select_discover_card(self, discoverables: List['MonsterCard']) -> 'MonsterCard':
        return self.rand.choice(discoverables)

    def select_from_store(self, store: List['MonsterCard']) -> 'MonsterCard':
        return self.rand.choice(store)

    def select_gain_card(self, cards: List['MonsterCard']) -> 'MonsterCard':
        return self.rand.choice(cards)

    def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
        return self.rand.choice(hero_pool)

    def select_summon_minion(self, cards: List['Type']) -> 'Type':
        return self.rand.choice(cards)

    def select_add_to_store(self, cards: List['MonsterCard']) -> 'MonsterCard':
        return self.rand.choice(cards)

    def select_monster_type(self, monster_types: List['MONSTER_TYPES'], round_number: int) -> 'MONSTER_TYPES':
        return self.rand.choice(monster_types)

    def select_random_minion(self, cards: List['Type'], round_number: int) -> 'Type':
        return self.rand.choice(cards)

    def select_adaptation(self, adaptations: List['Type']) -> 'Type':
        return self.rand.choice(adaptations)

    def select_random_number(self, lo: int, hi: int) -> int:
        return self.rand.randint(lo, hi)

    def select_secret(self, secrets: List['SECRETS']) -> 'SECRETS':
        return self.rand.choice(secrets)

    def select_combat_matchup(self, pairings: List[Tuple['Player', 'Player']]) -> Tuple['Player', 'Player']:
        return self.rand.choice(pairings)
