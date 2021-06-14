import typing
from typing import Optional, List

from hearthstone.simulator.core.cards import CardLocation

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.events import BuyPhaseContext
    from hearthstone.simulator.core.player import BoardIndex, StoreIndex


class Spell:
    base_cost: int = 0
    target_location: List['CardLocation'] = []
    darkmoon_prize_tier: int = 0

    def __init__(self, tier: Optional[int] = None):
        self.cost = self.base_cost
        self.tier = tier

    def __repr__(self):
        rep = f"{type(self).__name__}"
        rep += f"({self.cost})"
        if self.tier is not None:
            rep += f", [tier {self.tier}]"
        return "{" + rep + "}"

    def valid(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
              store_index: Optional['StoreIndex'] = None) -> bool:
        if self.target_location == []:
            return (board_index is None) and (store_index is None)
        if board_index is None and store_index is None:
            return False
        if board_index is not None:
            if CardLocation.BOARD not in self.target_location or not context.owner.valid_board_index(
                    board_index):
                return False
        if store_index is not None:
            if CardLocation.STORE not in self.target_location or not context.owner.valid_store_index(
                    store_index):
                return False
        if not self.valid_target(context, board_index, store_index):
            return False
        return True

    def valid_target(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                     store_index: Optional['StoreIndex'] = None):
        return True

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        pass

    def on_gain(self, context: 'BuyPhaseContext'):
        pass
