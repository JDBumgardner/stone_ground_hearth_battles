import typing
from typing import Union, Tuple, Optional, List, Any

from hearthstone.simulator.core.cards import CardLocation
from hearthstone.simulator.core.events import BuyPhaseContext, CombatPhaseContext, CardEvent, EVENTS
from hearthstone.simulator.core.monster_types import MONSTER_TYPES

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.player import BoardIndex, StoreIndex, DiscoverIndex, Player


class Hero:
    base_power_cost: Optional[int] = None  # default value is for heroes with passive hero powers
    hero_powers_per_turn = 1
    can_use_power = True
    power_target_location: Optional[List['CardLocation']] = None
    pool: 'MONSTER_TYPES' = MONSTER_TYPES.ALL

    def __init__(self):
        self.power_cost = self.base_power_cost
        self.discover_queue: List[List[Any]] = []
        self.give_immunity = False
        self.power_uses_this_turn = 0

    def __repr__(self):
        return str(type(self).__name__)

    def starting_health(self) -> int:
        return 40

    def minion_cost(self) -> int:
        return 3

    def refresh_cost(self) -> int:
        return 1

    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return 0, 5, 7, 8, 9, 10

    def occupied_store_slots(self) -> int:
        return 0

    def occupied_hand_slots(self) -> int:
        return 0

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END:
            self.give_immunity = False
        self.handle_event_powers(event, context)

    def handle_event_powers(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        pass

    def hero_power(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                   store_index: Optional['StoreIndex'] = None):
        assert self.hero_power_valid(context, board_index, store_index)
        context.owner.coins -= self.power_cost
        self.power_uses_this_turn += 1
        self.hero_power_impl(context, board_index, store_index)

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        pass

    def hero_power_valid(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                         store_index: Optional['StoreIndex'] = None):
        if self.power_cost is None:
            return False
        if context.owner.coins < self.power_cost:
            return False
        if self.power_uses_this_turn == self.hero_powers_per_turn:
            return False
        if not self.can_use_power:
            return False
        if self.power_target_location is None and (board_index is not None or store_index is not None):
            return False
        if self.power_target_location is not None:
            if board_index is None and store_index is None:
                return False
            if board_index is not None:
                if CardLocation.BOARD not in self.power_target_location or not context.owner.valid_board_index(
                        board_index):
                    return False
            if store_index is not None:
                if CardLocation.STORE not in self.power_target_location or not context.owner.valid_store_index(
                        store_index):
                    return False
        if not self.hero_power_valid_impl(context, board_index, store_index):
            return False
        return True

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return True

    def on_buy_step(self):
        self.power_uses_this_turn = 0

    def battlecry_multiplier(self) -> int:
        return 1

    def hero_info(self, player: 'Player') -> Optional[str]:
        return None

    def can_use_power_this_turn(self) -> bool:
        return self.power_uses_this_turn > 0


class EmptyHero(Hero):
    pass
