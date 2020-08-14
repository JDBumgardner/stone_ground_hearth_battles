import typing
from typing import Callable

from hearthstone.card_pool import *

if typing.TYPE_CHECKING:
    from hearthstone.cards import MonsterCard
    from hearthstone.player import Player


def attack_health_priority_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    return priority_function_bot(None, lambda player, card: card.health + card.attack + card.tier, seed)


def attack_health_tripler_priority_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        score = card.health + card.attack + card.tier
        num_existing = len([existing for existing in player.hand + player.in_play if
                            type(existing) == type(card) and not existing.golden])
        if num_existing == 2:
            score += 50
        elif num_existing == 1:
            score += 3

        return score

    return priority_function_bot(["Jake Bumgardner"], priority, seed)


def racist_priority_bot(seed: int, priority_function_bot: Callable, monster_type: MONSTER_TYPES):
    def priority(player: 'Player', card: 'MonsterCard'):
        score = card.health + card.attack + card.tier
        if card.monster_type == monster_type:
            score += 2
        return score

    return priority_function_bot(None, priority, seed)


def priority_saurolisk_bot(seed: int, priority_function_bot: Callable, monster_type: MONSTER_TYPES = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        if type(card) is RabidSaurolisk:
            return 100

        score = card.health + card.attack + card.tier
        if card.deathrattles:
            score += 5
        return score

    return priority_function_bot(["Jake Bumgardner"], priority, seed)


def priority_saurolisk_buff_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        if type(card) is RabidSaurolisk:
            return 100

        score = card.health + card.attack + card.tier
        rs_on_board = [card for card in player.in_play if type(card) is RabidSaurolisk]
        if rs_on_board and card.deathrattles:
            if card in player.hand + player.store:
                score += 9
        if rs_on_board and not card.deathrattles:
            score = -1
        return score

    return priority_function_bot(["Adam Salwen"], priority, seed)


def priority_adaptive_tripler_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        score = card.health + card.attack + card.tier
        num_existing = len([existing for existing in player.hand + player.in_play if
                            type(existing) == type(card) and not existing.golden])
        if num_existing == 2:
            score += 50
        elif num_existing == 1:
            score += 3

        counts = {}
        for existing in player.hand + player.in_play:
            counts[existing.monster_type] = counts.setdefault(existing.monster_type, 0) + 1
        score += counts.setdefault(card.monster_type, 0)

        return score

    return priority_function_bot(["Jeremy Salwen"], priority, seed)


def priority_health_tripler_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        score = card.health * 2 + card.attack + card.tier
        num_existing = len([existing for existing in player.hand + player.in_play if
                            type(existing) == type(card) and not existing.golden])
        if num_existing == 2:
            score += 50
        elif num_existing == 1:
            score += 3

        return score

    return priority_function_bot(["Jeremy Salwen"], priority, seed)


def priority_attack_tripler_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        score = card.health + card.attack * 2 + card.tier
        num_existing = len([existing for existing in player.hand + player.in_play if
                            type(existing) == type(card) and not existing.golden])
        if num_existing == 2:
            score += 50
        elif num_existing == 1:
            score += 3

        return score

    return priority_function_bot(["Jeremy Salwen"], priority, seed)


def battlerattler_priority_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        score = card.health + card.attack + card.tier
        num_existing = len([existing for existing in player.hand + player.in_play if
                            type(existing) == type(card) and not existing.golden])
        if num_existing == 2:
            score += 50
        elif num_existing == 1:
            score += 3

        counts = {}
        for existing in player.hand + player.in_play:
            counts[existing.monster_type] = counts.setdefault(existing.monster_type, 0) + 1
        score += counts.setdefault(card.monster_type, 0)

        if card.deathrattles:
            score += 2
        if card.battlecry:
            score += 2
        return score

    return priority_function_bot(["Jake Bumgardner"], priority, seed)


def priority_pogo_hopper_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        if type(card) is PogoHopper:
            return 100

        score = card.health + card.attack + card.tier

        return score

    return priority_function_bot(["Ethan Saxenian"], priority, seed)


def priority_jessie_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
    def priority(player: 'Player', card: 'MonsterCard'):
        if type(card) is WrathWeaver:
            return 100

        score = card.health + card.attack + card.tier
        ww_on_board = [card for card in player.in_play if type(card) is WrathWeaver]
        number_ww = len(ww_on_board)
        if number_ww and number_ww < player.health and card.monster_type is MONSTER_TYPES.DEMON:
            if card in player.hand + player.store:
                score += 9
        if ww_on_board and card.monster_type is not MONSTER_TYPES.DEMON:
            score = -1
        return score

    return priority_function_bot(["Adam Salwen"], priority, seed)