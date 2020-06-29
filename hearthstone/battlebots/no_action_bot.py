from typing import List

from hearthstone.agent import Agent, Action, EndPhaseAction
from hearthstone.cards import Card
from hearthstone.player import Player


class NoActionBot(Agent):
    def rearrange_cards(self, player: Player) -> List[Card]:
        return []

    def buy_phase_action(self, player: Player) -> Action:
        return EndPhaseAction(True)

    def discover_choice_action(self, player: Player) -> Card:
        return player.discovered_cards[0]
