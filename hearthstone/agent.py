import itertools
import typing
from typing import List, Optional, Generator

from hearthstone.monster_types import MONSTER_TYPES
from hearthstone.player import StoreIndex, HandIndex, BoardIndex

if typing.TYPE_CHECKING:
    from hearthstone.cards import Card
    from hearthstone.hero import Hero
    from hearthstone.tavern import Player


class Action:
    def apply(self, player: 'Player'):
        pass

    def valid(self, player: 'Player') -> bool:
        return False

    def str_in_context(self, player: 'Player') -> str:
        return str(self)


class BuyAction(Action):

    def __init__(self, index: StoreIndex):
        self.index = index

    def __repr__(self):
        return f"Buy({self.index})"

    def apply(self, player: 'Player'):
        player.purchase(self.index)

    def valid(self, player: 'Player'):
        return player.valid_purchase(self.index)

    def str_in_context(self, player: 'Player') -> str:
        return f"Buy({player.store[self.index]})"


class SummonAction(Action):
    def __init__(self, index: HandIndex, targets: Optional[List[BoardIndex]] = None):
        if targets is None:
            targets = []
        self.index = index
        self.targets = targets

    def __repr__(self):
        return f"Summon({self.index}, {self.targets})"

    def apply(self, player: 'Player'):
        player.summon_from_hand(self.index, self.targets)

    def valid(self, player: 'Player') -> bool:
        return player.valid_summon_from_hand(self.index, self.targets)

    def str_in_context(self, player: 'Player') -> str:
        return f"Summon({player.hand[self.index]},{self.targets})"


class SellAction(Action):

    def __init__(self, index: BoardIndex):
        self.index: BoardIndex = index

    def __repr__(self):
        return f"Sell({self.index})"

    def apply(self, player: 'Player'):
        player.sell_minion(self.index)

    def valid(self, player: 'Player') -> bool:
        return player.valid_sell_minion(self.index)

    def str_in_context(self, player: 'Player') -> str:
        return f"Sell({player.in_play[self.index]})"


class EndPhaseAction(Action):

    def __init__(self, freeze: bool):
        self.freeze: bool = freeze

    def __repr__(self):
        return f"EndPhase({self.freeze})"

    def apply(self, player: 'Player'):
        if self.freeze:
            player.freeze()

    def valid(self, player: 'Player') -> bool:
        return True


class RerollAction(Action):
    def __repr__(self):
        return f"Reroll()"

    def apply(self, player: 'Player'):
        player.reroll_store()

    def valid(self, player: 'Player') -> bool:
        return player.valid_reroll()


class TavernUpgradeAction(Action):
    def __repr__(self):
        return f"TavernUpgrade()"

    def apply(self, player: 'Player'):
        player.upgrade_tavern()

    def valid(self, player: 'Player') -> bool:
        return player.valid_upgrade_tavern()

    def str_in_context(self, player: 'Player') -> str:
        return f"TavernUpgrade({player.tavern_tier}, {player.tavern_upgrade_cost})"


class HeroPowerAction(Action):
    def __init__(self, board_target: Optional['BoardIndex'] = None, store_target: Optional['StoreIndex'] = None):
        self.board_target = board_target
        self.store_target = store_target

    def __repr__(self):
        return f"HeroPower({self.board_target}, {self.store_target})"

    def apply(self, player: 'Player'):
        player.hero_power(self.board_target, self.store_target)

    def valid(self, player: 'Player') -> bool:
        return player.valid_hero_power(self.board_target, self.store_target)


class TripleRewardsAction(Action):
    def __repr__(self):
        return f"TripleRewards()"

    def apply(self, player: 'Player'):
        player.play_triple_rewards()

    def valid(self, player: 'Player') -> bool:
        return player.valid_triple_rewards()

    def str_in_context(self, player: 'Player') -> str:
        return f"TripleRewards({player.triple_rewards[-1]})"


class RedeemGoldCoinAction(Action):
    def __repr__(self):
        return f"RedeemGoldCoin()"

    def apply(self, player: 'Player'):
        player.redeem_gold_coin()

    def valid(self, player: 'Player') -> bool:
        return player.gold_coins >= 1


class Agent:
    async def hero_choice_action(self, player: 'Player') -> 'Hero':
        return player.hero_options[0]

    async def rearrange_cards(self, player: 'Player') -> List['Card']:
        """
        here the player selects a card arangement one time per combat directly preceeding combat

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns: An arrangement of the player's board

        """
        pass

    async def buy_phase_action(self, player: 'Player') -> Action:
        """
        here the player chooses a buy phase action including:
        purchasing a card from the store
        summoning a card from hand to in_play
        selling a card from hand or from in_play
        and ending the buy phase

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns: one of four action types
        (BuyAction, SummonAction, SellAction, EndPhaseAction)

        """
        pass

    async def discover_choice_action(self, player: 'Player') -> 'Card':
        """

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:

        """
        pass

    async def game_over(self, player: 'Player', ranking: int):
        """
        Notifies the agent that the game is over and the agent has achieved a given rank
        :param ranking: Integer index 0 to 7 of where the agent placed
        :return:
        """
        pass


def generate_valid_actions(player: 'Player') -> Generator[Action, None, None]:
    return (action for action in generate_all_actions(player) if action.valid(player))


def generate_all_actions(player: 'Player') -> Generator[Action, None, None]:
    yield TripleRewardsAction()
    yield TavernUpgradeAction()
    yield RerollAction()
    yield EndPhaseAction(True)
    yield EndPhaseAction(False)
    yield HeroPowerAction()
    yield RedeemGoldCoinAction()
    for index in range(len(player.in_play)):
        yield SellAction(BoardIndex(index))
        yield HeroPowerAction(board_target=BoardIndex(index))
    for index in range(len(player.store)):
        yield BuyAction(StoreIndex(index))
        yield HeroPowerAction(store_target=StoreIndex(index))
    for index, card in enumerate(player.hand):
        valid_target_indices = [index for index, target in enumerate(player.in_play) if
                                card.valid_battlecry_target(target)]
        possible_num_targets = [num_targets for num_targets in card.num_battlecry_targets if
                                num_targets <= len(valid_target_indices)]
        if not possible_num_targets:
            possible_num_targets = [len(valid_target_indices)]
        for num_targets in possible_num_targets:
            for targets in itertools.combinations(valid_target_indices, num_targets):
                yield SummonAction(index, list(targets))
        if card.magnetic:
            for target_index, target_card in enumerate(player.in_play):
                if target_card.check_type(MONSTER_TYPES.MECH):
                    yield SummonAction(index, [target_index])
