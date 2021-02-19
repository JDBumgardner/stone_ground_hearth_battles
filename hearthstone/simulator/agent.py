import enum
import itertools
import typing
from typing import Any, List, Optional, Generator

import torch

from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.player import StoreIndex, HandIndex, BoardIndex, HeroChoiceIndex, DiscoverIndex


from hearthstone.simulator.core.tavern import Player


class FreezeDecision(enum.Enum):
    FREEZE = 0
    UNFREEZE = 1


class Action:
    def apply(self, player: 'Player'):
        pass

    def valid(self, player: 'Player') -> bool:
        return False

    def str_in_context(self, player: 'Player') -> str:
        return str(self)


class HeroChoiceAction(Action):
    def __init__(self, hero_index: HeroChoiceIndex):
        self.hero_index = hero_index

    def __repr__(self):
        return f"ChooseHero({self.hero_index})"

    def apply(self, player: 'Player'):
        player.choose_hero(self.hero_index)

    def valid(self, player: 'Player') -> bool:
        return player.valid_choose_hero(self.hero_index)

    def str_in_context(self, player: 'Player') -> str:
        return f"ChooseHero({player.hero_options[self.hero_index]})"


class DiscoverChoiceAction(Action):
    def __init__(self, card_index: 'DiscoverIndex'):
        self.card_index = card_index

    def __repr__(self):
        return f"Discover({self.card_index})"

    def apply(self, player: 'Player'):
        player.select_discover(self.card_index)

    def valid(self, player: 'Player') -> bool:
        return player.valid_select_discover(self.card_index)

    def str_in_context(self, player: 'Player') -> str:
        return f"Discover({player.discover_queue[0][self.card_index]})"


class RearrangeCardsAction(Action):
    def __init__(self, permutation: List[int]):
        self.permutation = permutation

    def __repr__(self):
        return f"Rearrange_cards({','.join([str(i) for i in self.permutation])})"

    def apply(self, player: 'Player'):
        player.rearrange_cards(self.permutation)

    def valid(self, player: 'Player') -> bool:
        return player.valid_rearrange_cards(self.permutation)


class HeroDiscoverAction(Action):
    def __init__(self, discover_index: 'DiscoverIndex'):
        self.discover_index = discover_index

    def __repr__(self):
        return f"Choose({self.discover_index})"

    def apply(self, player: 'Player'):
        player.hero_select_discover(self.discover_index)

    def valid(self, player: 'Player') -> bool:
        return player.valid_hero_select_discover(self.discover_index)

    def str_in_context(self, player: 'Player') -> str:
        return f"Choose({player.hero.discover_choices[self.discover_index]})"


class StandardAction(Action):
    pass


class BuyAction(StandardAction):

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


class SummonAction(StandardAction):
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


class SellAction(StandardAction):

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


class EndPhaseAction(StandardAction):

    def __init__(self, freeze: Optional[FreezeDecision] = None):
        self.freeze: Optional[FreezeDecision] = freeze

    def __repr__(self):
        return f"EndPhase({self.freeze.name if self.freeze is not None else ''})"

    def apply(self, player: 'Player'):
        if self.freeze == FreezeDecision.FREEZE:
            player.freeze()
        elif self.freeze == FreezeDecision.UNFREEZE:
            player.unfreeze()

    def valid(self, player: 'Player') -> bool:
        if self.freeze == FreezeDecision.UNFREEZE and not any(card.frozen for card in player.store):
            return False
        return not player.dead and not player.discover_queue


class RerollAction(StandardAction):
    def __repr__(self):
        return f"Reroll()"

    def apply(self, player: 'Player'):
        player.reroll_store()

    def valid(self, player: 'Player') -> bool:
        return player.valid_reroll()


class TavernUpgradeAction(StandardAction):
    def __repr__(self):
        return f"TavernUpgrade()"

    def apply(self, player: 'Player'):
        player.upgrade_tavern()

    def valid(self, player: 'Player') -> bool:
        return player.valid_upgrade_tavern()

    def str_in_context(self, player: 'Player') -> str:
        return f"TavernUpgrade({player.tavern_tier}, {player.tavern_upgrade_cost})"


class HeroPowerAction(StandardAction):
    def __init__(self, board_target: Optional['BoardIndex'] = None, store_target: Optional['StoreIndex'] = None):
        self.board_target = board_target
        self.store_target = store_target

    def __repr__(self):
        return f"HeroPower({self.board_target}, {self.store_target})"

    def apply(self, player: 'Player'):
        player.hero_power(self.board_target, self.store_target)

    def valid(self, player: 'Player') -> bool:
        return player.valid_hero_power(self.board_target, self.store_target)


class TripleRewardsAction(StandardAction):
    def __repr__(self):
        return f"TripleRewards()"

    def apply(self, player: 'Player'):
        player.play_triple_rewards()

    def valid(self, player: 'Player') -> bool:
        return player.valid_triple_rewards()

    def str_in_context(self, player: 'Player') -> str:
        return f"TripleRewards({player.triple_rewards[-1]})"


class RedeemGoldCoinAction(StandardAction):
    def __repr__(self):
        return f"RedeemGoldCoin()"

    def apply(self, player: 'Player'):
        player.redeem_gold_coin()

    def valid(self, player: 'Player') -> bool:
        return player.gold_coins >= 1 and not player.dead


class BananaAction(StandardAction):
    def __init__(self, board_target: Optional['BoardIndex'] = None, store_target: Optional['StoreIndex'] = None):
        self.board_target = board_target
        self.store_target = store_target

    def __repr__(self):
        return f"Banana({self.board_target}, {self.store_target})"

    def apply(self, player: 'Player'):
        player.use_banana(self.board_target, self.store_target)

    def valid(self, player: 'Player') -> bool:
        return player.valid_use_banana(self.board_target, self.store_target)


Annotation = Any


class AnnotatingAgent:
    async def hero_choice_action(self, player: 'Player') -> HeroChoiceAction:
        return HeroChoiceAction(HeroChoiceIndex(0))

    async def annotated_rearrange_cards(self, player: 'Player') -> (RearrangeCardsAction, Annotation):
        """
        here the player selects a card arrangement one time per combat directly preceding combat

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:  A tuple containing an arrangement of the player's board, and the Agent Annotation to attach to the replay.

        """
        pass

    async def annotated_buy_phase_action(self, player: 'Player') -> (StandardAction, Annotation):
        """
        here the player chooses a buy phase action including:
        purchasing a card from the store
        summoning a card from hand to in_play
        selling a card from hand or from in_play
        and ending the buy phase

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:
            A tuple containing the Action, and the Agent Annotation to attach to the replay.

        """
        pass

    async def annotated_discover_choice_action(self, player: 'Player') -> (DiscoverChoiceAction, Annotation):
        """

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:
            Tuple of MonsterCard to discover, and Annotation to attach to the action.
        """
        pass

    async def annotated_hero_discover_action(self, player: 'Player') -> ('HeroDiscoverAction', Annotation):
        """

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:
            Tuple of object to discover, and Annotation to attach to the action.
        """
        pass

    async def game_over(self, player: 'Player', ranking: int) -> Annotation:
        """
        Notifies the agent that the game is over and the agent has achieved a given rank
        :param ranking: Integer index 0 to 7 of where the agent placed
        :return:
        """
        pass


class Agent(AnnotatingAgent):

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        pass

    async def annotated_buy_phase_action(self, player: 'Player') -> (StandardAction, Annotation):
        return await self.buy_phase_action(player), None

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        pass

    async def annotated_rearrange_cards(self, player: 'Player') -> (RearrangeCardsAction, Annotation):
        return await self.rearrange_cards(player), None

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        pass

    async def annotated_discover_choice_action(self, player: 'Player') -> (DiscoverChoiceAction, Annotation):
        return await self.discover_choice_action(player), None

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        pass

    async def annotated_hero_discover_action(self, player: 'Player') -> ('HeroDiscoverAction', Annotation):
        return await self.hero_discover_action(player), None


def generate_valid_actions(player: 'Player') -> Generator[StandardAction, None, None]:
    return (action for action in generate_all_actions(player) if action.valid(player))


def generate_all_actions(player: 'Player') -> Generator[StandardAction, None, None]:
    yield TripleRewardsAction()
    yield TavernUpgradeAction()
    yield RerollAction()
    yield EndPhaseAction()
    yield EndPhaseAction(FreezeDecision.FREEZE)
    yield EndPhaseAction(FreezeDecision.UNFREEZE)
    yield HeroPowerAction()
    yield RedeemGoldCoinAction()
    for index in range(len(player.in_play)):
        yield SellAction(BoardIndex(index))
        yield HeroPowerAction(board_target=BoardIndex(index))
        yield BananaAction(board_target=BoardIndex(index))
    for index in range(len(player.store)):
        yield BuyAction(StoreIndex(index))
        yield HeroPowerAction(store_target=StoreIndex(index))
        yield BananaAction(store_target=StoreIndex(index))
    for index, card in enumerate(player.hand):
        valid_target_indices = [index for index, target in enumerate(player.in_play) if
                                card.valid_battlecry_target(target)]
        possible_num_targets = [num_targets for num_targets in card.num_battlecry_targets if
                                num_targets <= len(valid_target_indices)]
        if not possible_num_targets:
            possible_num_targets = [len(valid_target_indices)]
        for num_targets in possible_num_targets:
            for targets in itertools.combinations(valid_target_indices, num_targets):
                yield SummonAction(HandIndex(index), [BoardIndex(target_index) for target_index in targets])
        if card.magnetic:
            for target_index, target_card in enumerate(player.in_play):
                if target_card.check_type(MONSTER_TYPES.MECH):
                    yield SummonAction(HandIndex(index), [BoardIndex(target_index)])
