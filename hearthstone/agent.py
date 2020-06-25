from typing import List, Optional

from hearthstone.cards import Card, MonsterCard
from hearthstone.tavern import Player


class Action:
    def apply(self, player: Player):
        pass


class BuyAction(Action):

    def __init__(self, card):
        self.card: MonsterCard = card

    def apply(self, player: Player):
        assert self.card in player.store
        player.purchase(player.store.index(self.card))


class SummonAction(Action):

    def __init__(self, card: MonsterCard, target_1: MonsterCard, target_2: MonsterCard):
        self.card: MonsterCard = card
        self.target_1: Optional[MonsterCard] = target_1
        self.target_2: Optional[MonsterCard] = target_2

    def apply(self, player: Player):
        assert self.card in player.hand
        if self.target_1 not in player.in_play:
            self.target_1 = None
        if self.target_2 not in player.in_play:
            self.target_2 = None
        player.summon_from_hand(self.card, self.target_1, self.target_2)


class SellAction(Action):

    def __init__(self, card: MonsterCard):
        self.card: MonsterCard = card

    def apply(self, player: Player):
        assert self.card in player.hand or self.card in player.in_play
        player.sell_minion(self.card)

class EndPhaseAction(Action):

    def __init__(self, freeze: bool):
        self.freeze: bool = freeze

    def apply(self, player: Player):
        if self.freeze:
            player.freeze()


class RerollAction(Action):

    def apply(self, player: Player):
        player.refresh_store()


class TavernUpgradeAction(Action):

    def apply(self, player: Player):
        player.upgrade_tavern()


class HeroPowerAction(Action):
    def apply(self, player: Player):
        player.hero_power()


class TripleRewardAction(Action):
    def apply(self, player: Player):
        player.generic_discover()


class Agent:
    def rearrange_cards(self, player: Player) -> List[Card]:
        """
        here the player selects a card arangement one time per combat directly preceeding combat

        Args:
            player: The player object coutrolled by this agent. This function should not modify it.

        Returns: An arrangement of the player's board

        """
        pass

    def buy_phase_action(self, player: Player) -> Action:
        """
        here the player chooses a buy phase action including:
        purchasing a card from the store
        summoning a card from hand to in_play
        selling a card from hand or from in_play
        and ending the buy phase

        Args:
            player: The player object coutrolled by this agent. This function should not modify it.

        Returns: one of four action types
        (BuyAction, SummonAction, SellAction, EndPhaseAction)

        """
        pass

    def discover_choice_action(self, player: Player) -> Card:
        """

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:

        """
        pass
