import random
from typing import List

from hearthstone.agent import Agent, Action, generate_valid_actions, BuyAction, EndPhaseAction, SummonAction, \
    TavernUpgradeAction
from hearthstone.cards import Card
from hearthstone.player import Player


class SupremacyBot(Agent):
    authors = ["Jeremy Salwen"]

    def __init__(self, monster_type:str, upgrade: bool, seed: int):
        self.local_random = random.Random(seed)
        self.monster_type = monster_type
        self.upgrade = upgrade

    def rearrange_cards(self, player: Player) -> List[Card]:
        card_list = player.in_play.copy()
        self.local_random.shuffle(card_list)
        return card_list

    def buy_phase_action(self, player: Player) -> Action:
        all_actions = list(generate_valid_actions(player))

        if self.upgrade:
            upgrade_actions = [action for action in all_actions if type(action) is TavernUpgradeAction]
            if upgrade_actions:
                return upgrade_actions[0]

        summon_actions = [action for action in all_actions if type(action) is SummonAction]
        if summon_actions:
            return summon_actions[0]

        buy_actions = [action for action in all_actions if type(action) is BuyAction and action.card.monster_type == self.monster_type]
        buy_actions = sorted(buy_actions, key=lambda buy_action: buy_action.card.tier, reverse=True)
        if buy_actions:
            return buy_actions[0]

        return EndPhaseAction(False)

    def discover_choice_action(self, player: Player) -> Card:
        discover_cards = player.discovered_cards
        discover_cards = sorted(discover_cards, key=lambda card: card.tier, reverse=True)
        return discover_cards[0]
