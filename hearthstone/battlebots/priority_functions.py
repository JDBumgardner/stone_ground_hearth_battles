import typing
from typing import Callable

from hearthstone.simulator.core.card_pool import *

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.cards import MonsterCard
    from hearthstone.simulator.core.player import Player


class PriorityFunctions:
    @staticmethod
    def attack_health_priority_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
        return priority_function_bot(None, lambda player, card: card.health + card.attack + card.tier, seed)

    @staticmethod
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

    @staticmethod
    def racist_priority_bot(seed: int, priority_function_bot: Callable, monster_type: MONSTER_TYPES):
        def priority(player: 'Player', card: 'MonsterCard'):
            score = card.health + card.attack + card.tier
            if card.monster_type == monster_type:
                score += 2
            return score

        return priority_function_bot(None, priority, seed)

    @staticmethod
    def priority_saurolisk_bot(seed: int, priority_function_bot: Callable, monster_type: MONSTER_TYPES = None):
        def priority(player: 'Player', card: 'MonsterCard'):
            if type(card) is RabidSaurolisk:
                return 100

            score = card.health + card.attack + card.tier
            if card.deathrattles:
                score += 5
            return score

        return priority_function_bot(["Jake Bumgardner"], priority, seed)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def priority_pack_leader_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
        def priority(player: 'Player', card: 'MonsterCard'):
            if type(card) is PackLeader:
                return 20

            score = card.health + card.attack + card.tier
            pack_leader_on_board = [card for card in player.in_play if type(card) is PackLeader]
            if pack_leader_on_board and card.monster_type is MONSTER_TYPES.BEAST:
                score += 10
                if card.deathrattles:
                    score += 5
            return score

        return priority_function_bot(["Ethan Saxenian"], priority, seed)

    @staticmethod
    def priority_togwaggle_bot(seed: int, priority_function_bot: Callable, monster_type: str = None):
        def priority(player: 'Player', card: 'MonsterCard'):
            if type(card) is WaxriderTogwaggle:
                return 20

            score = card.health + card.attack + card.tier
            togwaggle_on_board = [card for card in player.in_play if type(card) is WaxriderTogwaggle]
            if togwaggle_on_board and card.monster_type is MONSTER_TYPES.DRAGON:
                score += 10
            return score

        return priority_function_bot(["Ethan Saxenian"], priority, seed)

    @staticmethod
    def priority_callables_bot(seed: int, priority_function_bot: Callable, monster_type: str = None,
                               modifiers_list: List[Callable] = None):
        def priority(player: 'Player', card: 'MonsterCard'):
            modifier = 0
            for modifier_function in modifiers_list:
                modifier += modifier_function.get_modification(player, card)
            score = card.health + card.attack + card.tier + modifier
            return score

        return priority_function_bot(["Jacob Bumgardner"], priority, seed)