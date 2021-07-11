import enum
import itertools
from typing import List, Optional, Generator

import autoslot

from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.player import HeroChoiceIndex, StoreIndex, HandIndex, BoardIndex, SpellIndex, Player, \
    DiscoverIndex


class FreezeDecision(enum.Enum):
    NO_FREEZE = 0
    FREEZE = 1
    UNFREEZE = 2


class Action(autoslot.Slots):
    def apply(self, player: 'Player'):
        pass

    def valid(self, player: 'Player') -> bool:
        return False

    def base_valid(self, player: 'Player') -> bool:
        return False

    def str_in_context(self, player: 'Player') -> str:
        return str(self)


class HeroChoiceAction(Action):
    def __init__(self, hero_index: HeroChoiceIndex):
        self.hero_index = hero_index

    def __repr__(self):
        return f"ChooseHero({self.hero_index})"

    def apply(self, player: 'Player'):
        player.choose_hero_from_index(self.hero_index)

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


class StandardAction(Action):
    def valid(self, player: 'Player') -> bool:
        return player.valid_standard_action() and self.base_valid(player)


class BuyAction(StandardAction):
    def __init__(self, index: StoreIndex):
        self.index = index

    def __repr__(self):
        return f"Buy({self.index})"

    def apply(self, player: 'Player'):
        player.purchase(self.index)

    def base_valid(self, player: 'Player') -> bool:
        return player.base_valid_purchase(self.index)

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

    def base_valid(self, player: 'Player') -> bool:
        return player.base_valid_summon_from_hand(self.index, self.targets)

    def str_in_context(self, player: 'Player') -> str:
        return f"Summon({player.hand[self.index]},{self.targets})"


class SellAction(StandardAction):

    def __init__(self, index: BoardIndex):
        self.index: BoardIndex = index

    def __repr__(self):
        return f"Sell({self.index})"

    def apply(self, player: 'Player'):
        player.sell_minion(self.index)

    def base_valid(self, player: 'Player') -> bool:
        return player.base_valid_sell_minion(self.index)

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

    def base_valid(self, player: 'Player') -> bool:
        return self.freeze != FreezeDecision.UNFREEZE or any(card.frozen for card in player.store)


class RerollAction(StandardAction):
    def __repr__(self):
        return f"Reroll()"

    def apply(self, player: 'Player'):
        player.reroll_store()

    def base_valid(self, player: 'Player') -> bool:
        return player.base_valid_reroll()


class TavernUpgradeAction(StandardAction):
    def __repr__(self):
        return f"TavernUpgrade()"

    def apply(self, player: 'Player'):
        player.upgrade_tavern()

    def base_valid(self, player: 'Player') -> bool:
        return player.base_valid_upgrade_tavern()

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

    def base_valid(self, player: 'Player') -> bool:
        return player.base_valid_hero_power(self.board_target, self.store_target)


class PlaySpellAction(StandardAction):
    def __init__(self, index: 'SpellIndex', board_target: Optional['BoardIndex'] = None,
                 store_target: Optional['StoreIndex'] = None):
        self.index = index
        self.board_target = board_target
        self.store_target = store_target

    def __repr__(self):
        return f"PlaySpell({self.index}, [{self.board_target}, {self.store_target}])"

    def apply(self, player: 'Player'):
        player.play_spell(self.index, self.board_target, self.store_target)

    def base_valid(self, player: 'Player') -> bool:
        return player.base_valid_play_spell(self.index, self.board_target, self.store_target)


def yield_if_base_valid(player: 'Player', action: 'StandardAction') -> Generator[StandardAction, None, None]:
    if action.base_valid(player):
        yield action


def generate_standard_actions(player: 'Player') -> Generator[StandardAction, None, None]:
    if not player.valid_standard_action():
        return
    yield EndPhaseAction(FreezeDecision.NO_FREEZE)
    yield EndPhaseAction(FreezeDecision.FREEZE)
    yield from yield_if_base_valid(player, EndPhaseAction(FreezeDecision.UNFREEZE))

    yield from yield_if_base_valid(player, TavernUpgradeAction())
    yield from yield_if_base_valid(player, RerollAction())
    yield from yield_if_base_valid(player, HeroPowerAction())
    for index in range(len(player.in_play)):
        yield SellAction(BoardIndex(index))
        yield from yield_if_base_valid(player, HeroPowerAction(board_target=BoardIndex(index)))

    for index in range(len(player.store)):
        yield from yield_if_base_valid(player, BuyAction(StoreIndex(index)))
        yield from yield_if_base_valid(player, HeroPowerAction(store_target=StoreIndex(index)))

    for index in range(len(player.spells)):
        yield from yield_if_base_valid(player, PlaySpellAction(SpellIndex(index)))
        for board_index in range(len(player.in_play)):
            yield from yield_if_base_valid(player,
                                           PlaySpellAction(SpellIndex(index), board_target=BoardIndex(board_index)))

        for store_index in range(len(player.store)):
            yield from yield_if_base_valid(player,
                                           PlaySpellAction(SpellIndex(index), store_target=StoreIndex(store_index)))

    if player.room_on_board():
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
