import random
from typing import List, Callable

from hearthstone.agent import Agent, Action, generate_valid_actions, BuyAction, EndPhaseAction, SummonAction, \
    SellAction, TavernUpgradeAction, RerollAction
from hearthstone.card_pool import RabidSaurolisk
from hearthstone.cards import Card, MonsterCard
from hearthstone.player import Player


class PriorityBot(Agent):
    def __init__(self, authors: List[str], priority: Callable[[MonsterCard], float], seed: int):
        if not authors:
            authors = ["Jake Bumgardner", "Jeremy Salwen", "Diana Valverde-Paniagua"]
        self.authors = authors
        self.priority = priority
        self.local_random = random.Random(seed)

    def rearrange_cards(self, player: Player) -> List[Card]:
        card_list = player.in_play.copy()
        self.local_random.shuffle(card_list)
        return card_list

    def buy_phase_action(self, player: Player) -> Action:
        all_actions = list(generate_valid_actions(player))

        if player.tavern_tier < 2:
            upgrade_action = TavernUpgradeAction()
            if upgrade_action.valid(player):
                return upgrade_action

        top_hand_priority = max([self.priority(card) for card in player.hand], default=None)
        top_store_priority = max([self.priority(card) for card in player.store], default=None)
        bottom_board_priority = min([self.priority(card) for card in player.in_play], default=None)
        if top_hand_priority:
            if player.room_on_board():
                return [action for action in all_actions if type(action) is SummonAction and self.priority(action.card) == top_hand_priority][0]
            else:
                if top_hand_priority > bottom_board_priority:
                    return [action for action in all_actions if type(action) is SellAction and self.priority(action.card) == bottom_board_priority][0]

        if top_store_priority:
            if player.room_on_board() or bottom_board_priority < top_store_priority:
                buy_action = BuyAction([card for card in player.store if self.priority(card) == top_store_priority][0])
                if buy_action.valid(player):
                    return buy_action

        reroll_action = RerollAction()
        if reroll_action.valid(player):
            return reroll_action

        return EndPhaseAction(False)

    def discover_choice_action(self, player: Player) -> Card:
        discover_cards = player.discovered_cards
        discover_cards = sorted(discover_cards, key=lambda card: self.priority(card), reverse=True)
        return discover_cards[0]


def attack_health_priority_bot(seed: int):
    return PriorityBot(None, lambda card: card.health + card.attack + card.tier, seed)


def racist_priority_bot(monster_type: str, seed: int):
    def priority(card: MonsterCard):
        score = card.health + card.attack + card.tier
        if card.monster_type == monster_type:
            score += 2
        return score

    return PriorityBot(None, priority, seed)


def priority_saurolisk_bot(seed: int):
    def priority(card: MonsterCard):
        if type(card) is RabidSaurolisk:
            return 100

        score = card.health + card.attack + card.tier
        if card.deathrattles:
            score += 5
        return score

    return PriorityBot(["Jake Bumgardner"], priority, seed)
