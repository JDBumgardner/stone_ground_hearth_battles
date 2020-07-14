import random
import typing
from typing import List

from hearthstone.agent import Agent, Action, generate_valid_actions, BuyAction, EndPhaseAction, SummonAction, \
    SellFromBoardAction, SellFromHandAction, TavernUpgradeAction, RerollAction
from hearthstone.card_pool import RabidSaurolisk
if typing.TYPE_CHECKING:
    from hearthstone.cards import Card
    from hearthstone.player import Player, BoardIndex


class SauroliskBot(Agent):
    authors = ["Jake Bumgardner"]
    def __init__(self, seed: int):
        self.local_random = random.Random(seed)

    def rearrange_cards(self, player: 'Player') -> List['Card']:
        card_list = player.in_play.copy()
        self.local_random.shuffle(card_list)
        return card_list

    @staticmethod
    def desired_card(card):
        return type(card) == RabidSaurolisk or card.deathrattles

    def buy_phase_action(self, player: 'Player') -> Action:
        all_actions = list(generate_valid_actions(player))

        upgrade_actions = [action for action in all_actions if type(action) is TavernUpgradeAction]
        if upgrade_actions:
            return upgrade_actions[0]

        summon_actions = [action for action in all_actions if type(action) is SummonAction]
        if summon_actions:
            return summon_actions[0]

        buy_actions = [action for action in all_actions if type(action) is BuyAction and self.desired_card(action.card)]
        if buy_actions:
            return buy_actions[0]

        reroll_action = RerollAction()
        if reroll_action.valid(player):
            return reroll_action

        if len(player.in_play) == 7:
            for index, card in enumerate(player.in_play):
                if type(card) is not RabidSaurolisk:
                    return SellFromBoardAction(BoardIndex(index))

        return EndPhaseAction(False)

    def discover_choice_action(self, player: 'Player') -> 'Card':
        discover_cards = player.discovered_cards
        discover_cards = sorted(discover_cards, key=lambda card: self.desired_card(card), reverse=True)
        return discover_cards[0]
