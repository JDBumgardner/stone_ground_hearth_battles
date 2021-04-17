import enum
import itertools
from typing import List, Optional, Generator

from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.player import HeroChoiceIndex, StoreIndex, HandIndex, BoardIndex


class FreezeDecision(enum.Enum):
    NO_FREEZE = 0
    FREEZE = 1
    UNFREEZE = 2


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

    def __init__(self, freeze: FreezeDecision):
        self.freeze = freeze

    def __repr__(self):
        return f"EndPhase({self.freeze.name})"

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


def generate_valid_actions(player: 'Player') -> Generator[StandardAction, None, None]:
    return (action for action in generate_all_actions(player) if action.valid(player))


def generate_all_actions(player: 'Player') -> Generator[StandardAction, None, None]:
    yield TripleRewardsAction()
    yield TavernUpgradeAction()
    yield RerollAction()
    yield EndPhaseAction(FreezeDecision.NO_FREEZE)
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
        if card.num_battlecry_targets:
            valid_target_indices = [index for index, target in enumerate(player.in_play) if
                                    card.valid_battlecry_target(target)]
            possible_num_targets = [num_targets for num_targets in card.num_battlecry_targets if
                                    num_targets <= len(valid_target_indices)]
            if not possible_num_targets:
                possible_num_targets = [len(valid_target_indices)]
            for num_targets in possible_num_targets:
                for targets in itertools.combinations(valid_target_indices, num_targets):
                    yield SummonAction(HandIndex(index), [BoardIndex(target_index) for target_index in targets])
        else:
            yield SummonAction(HandIndex(index), [])
        if card.magnetic:
            for target_index, target_card in enumerate(player.in_play):
                if target_card.check_type(MONSTER_TYPES.MECH):
                    yield SummonAction(HandIndex(index), [BoardIndex(target_index)])