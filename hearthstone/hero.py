import typing
from typing import Union, Tuple, Optional

from hearthstone.card_factory import make_metaclass
from hearthstone.cards import CardLocation

from hearthstone.events import BuyPhaseContext, CombatPhaseContext, CardEvent

if typing.TYPE_CHECKING:
    from hearthstone.player import BoardIndex, StoreIndex

VALHALLA = []

HeroType = make_metaclass(VALHALLA.append, ("Hero", "EmptyHero"))


class Hero(metaclass=HeroType):
    power_cost: Optional[int] = None  # default value is for heroes with passive hero powers
    hero_power_used = False
    can_use_power = True
    power_target_location: Optional['CardLocation'] = None
    multiple_power_uses_per_turn = False

    def __repr__(self):
        return str(type(self).__name__)

    def starting_health(self) -> int:
        return 40

    def minion_cost(self) -> int:
        return 3

    def refresh_cost(self) -> int:
        return 1

    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return (0, 5, 7, 8, 9, 10)

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        pass

    def hero_power(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                   store_index: Optional['StoreIndex'] = None):
        assert self.hero_power_valid(context, board_index, store_index)
        context.owner.coins -= self.power_cost
        self.hero_power_used = True
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
        if not self.multiple_power_uses_per_turn:
            if self.hero_power_used:
                return False
        if not self.can_use_power:
            return False
        if board_index is None and store_index is None and self.power_target_location is not None:
            return False
        if board_index is not None:
            if self.power_target_location != CardLocation.BOARD or not context.owner.valid_board_index(board_index):
                return False
        if store_index is not None:
            if self.power_target_location != CardLocation.STORE or not context.owner.valid_store_index(store_index):
                return False
        if not self.hero_power_valid_impl(context, board_index, store_index):
            return False
        return True

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return True

    def on_buy_step(self):
        self.hero_power_used = False


class EmptyHero(Hero):
    pass
