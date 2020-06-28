from typing import List, Optional

from hearthstone.cards import Card, MonsterCard
from hearthstone.tavern import Player


class Action:
    def apply(self, player: Player):
        pass

    def valid(self, player: Player) -> bool:
        return False


class BuyAction(Action):

    def __init__(self, card):
        self.card: MonsterCard = card

    def apply(self, player: Player):
        assert self.card in player.store
        player.purchase(player.store.index(self.card))

    def valid(self, player: Player):
        if self.card not in player.store:
            return False
        if self.card.coin_cost > player.coins:
            return False
        if player.hand_size() >= player.maximum_hand_size:
            return False
        return True


class SummonAction(Action):

    def __init__(self, card: MonsterCard, targets: Optional[List[MonsterCard]] = None):
        if targets is None:
            targets = []
        self.card: MonsterCard = card
        self.targets = targets

    def apply(self, player: Player):
        assert self.card in player.hand
        player.summon_from_hand(self.card, self.targets)

    def valid(self, player: Player) -> bool:
        if self.card not in player.hand:
            return False
        if not player.room_on_board():
            return False
        if not self.card.validate_battlecry_target():
            return False
        # TODO: US do the targets thing
        print("you have to fix the summon targets thing!")


class SellAction(Action):

    def __init__(self, card: MonsterCard):
        self.card: MonsterCard = card

    def apply(self, player: Player):
        assert self.card in player.hand or self.card in player.in_play
        player.sell_minion(self.card)

    def valid(self, player: Player) -> bool:
        return self.card in player.hand + player.in_play


class EndPhaseAction(Action):

    def __init__(self, freeze: bool):
        self.freeze: bool = freeze

    def apply(self, player: Player):
        if self.freeze:
            player.freeze()

    def valid(self, player: Player) -> bool:
        return True


class RerollAction(Action):

    def apply(self, player: Player):
        player.refresh_store()

    def valid(self, player: Player) -> bool:
        return player.coins >= player.refresh_store_cost


class TavernUpgradeAction(Action):

    def apply(self, player: Player):
        player.upgrade_tavern()

    def valid(self, player: Player) -> bool:
        if player.tavern_tier >= player.max_tier():
            return False
        if player.coins < player.tavern_upgrade_cost():
            return False
        return True


class HeroPowerAction(Action):
    def apply(self, player: Player):
        player.hero_power()

    def valid(self, player: Player) -> bool:
        return player.hero_power_valid()


class TripleRewardAction(Action):
    def apply(self, player: Player):
        player.play_triple_rewards()

    def valid(self, player: Player) -> bool:
        return bool(player.triple_rewards)


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
