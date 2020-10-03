import random
import typing
from typing import List, Callable

from hearthstone.simulator.agent import Agent, Action, generate_valid_actions, BuyAction, EndPhaseAction, SummonAction, \
    TavernUpgradeAction, RerollAction, SellAction

if typing.TYPE_CHECKING:
    , MonsterCard
    from hearthstone.simulator.core.player import Player, StoreIndex


class PriorityStorageBot(Agent):
    def __init__(self, authors: List[str], priority: Callable[['Player', 'MonsterCard'], float], storage_priority: Callable[['Player', 'MonsterCard'], float], seed: int):
        if not authors:
            authors = ["Jacob Bumgardner", "Jeremy Salwen", "Diana Valverde"]
        self.authors = authors
        self.priority = priority
        self.storage_priority = storage_priority
        self.local_random = random.Random(seed)

    async def rearrange_cards(self, player: 'Player') -> List['MonsterCard']:
        card_list = player.in_play.copy()
        self.local_random.shuffle(card_list)
        return card_list

    async def buy_phase_action(self, player: 'Player') -> Action:

        all_actions = list(generate_valid_actions(player))

        if player.tavern_tier < 2:
            upgrade_action = TavernUpgradeAction()
            if upgrade_action.valid(player):
                return upgrade_action

        top_hand_priority = max([self.priority(player, card) for card in player.hand], default=None)
        top_store_priority = max([self.priority(player, card) for card in player.store], default=None)
        bottom_board_priority = min([self.priority(player, card) for card in player.in_play], default=None)

        if top_hand_priority:
            if player.room_on_board():
                return [action for action in all_actions if type(action) is SummonAction and self.priority(player, action.card) == top_hand_priority][0]
            else:
                if top_hand_priority > bottom_board_priority and player.coins >= 2:
                    return [action for action in all_actions if type(action) is SellAction and self.priority(player, player.in_play[action.index]) == bottom_board_priority][0]

        if top_store_priority:
            if player.room_in_hand():
                buy_action = BuyAction([StoreIndex(i) for i, card in enumerate(player.store) if self.priority(player, card) == top_store_priority][0])
                if buy_action.valid(player):
                    return buy_action
            elif bottom_board_priority < top_store_priority and player.coins >= 2:
                pass


        bottom_hand_storage_priority = min([self.storage_priority(player, card) for card in player.hand], default=None)
        top_store_storage_priority = max([self.storage_priority(player, card) for card in player.store], default=None)

        if top_store_storage_priority:
            if player.room_in_hand():
                buy_action = BuyAction([StoreIndex(i) for i, card in enumerate(player.store) if self.storage_priority(player, card) == top_store_storage_priority][0])
                if buy_action.valid(player):
                    return buy_action

        reroll_action = RerollAction()
        if reroll_action.valid(player):
            return reroll_action

        return EndPhaseAction(False)

    async def discover_choice_action(self, player: 'Player') -> 'MonsterCard':
        discover_cards = player.discover_queue[0]
        discover_cards = sorted(discover_cards, key=lambda card: self.priority(player, card), reverse=True)
        return discover_cards[0]


def priority_st_ad_tr_bot(seed: int):
    def priority(player: 'Player', card: 'MonsterCard'):
        score = card.health + card.attack + card.tier
        num_existing = len([existing for existing in player.hand + player.in_play if
                            type(existing) == type(card) and not existing.golden])
        if num_existing == 2:
            score += 50
        counts = {}
        for existing in player.hand + player.in_play:
            counts[existing.monster_type] = counts.setdefault(existing.monster_type, 0) + 1
        score += counts.setdefault(card.monster_type, 0)
        return score

    def storage_priority(player: 'Player', card: 'MonsterCard'):
        score = 0
        num_existing = len([existing for existing in player.hand + player.in_play if
                            type(existing) == type(card) and not existing.golden])
        if num_existing == 2:
            score += 50
        elif num_existing == 1:
            score += 50

        return score

    return PriorityStorageBot(["Jacob Bumgardner"], priority, storage_priority, seed)
