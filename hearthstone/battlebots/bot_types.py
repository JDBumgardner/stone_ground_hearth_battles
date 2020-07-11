import random
from typing import List, Callable

from hearthstone.agent import Agent, Action, generate_valid_actions, BuyAction, EndPhaseAction, SummonAction, \
    SellAction, TavernUpgradeAction, RerollAction
from hearthstone.card_pool import *
from hearthstone.cards import Card, MonsterCard
from hearthstone.player import Player


class PriorityFunctionBot(Agent):
    def __init__(self, authors: List[str], priority: Callable[[Player, MonsterCard], float], seed: int):
        if not authors:
            authors = ["JB", "AS", "ES", "JS", "DVP"]
        self.authors = authors
        self.priority = priority
        self.local_random = random.Random(seed)

    def discover_choice_action(self, player: Player) -> Card:
        discover_cards = player.discovered_cards
        discover_cards = sorted(discover_cards, key=lambda card: self.priority(card), reverse=True)
        return discover_cards[0]

    def rearrange_cards(self, player: Player) -> List[Card]:
        card_list = player.in_play.copy()
        self.local_random.shuffle(card_list)
        return card_list

    def buy_phase_action(self, player: Player) -> Action:
        pass
