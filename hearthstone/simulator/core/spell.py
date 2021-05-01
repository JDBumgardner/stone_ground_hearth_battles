from typing import Optional, List

from hearthstone.simulator.core.cards import CardLocation
from hearthstone.simulator.core.events import BuyPhaseContext
from hearthstone.simulator.core.player import BoardIndex, StoreIndex


class Spell:
    base_cost: int = 0
    target_location: Optional[List['CardLocation']] = None

    def __init__(self):
        self.cost = self.base_cost

    def __repr__(self):
        return type(self).__name__

    def valid(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None) -> bool:
        if context.owner.coins < self.cost:
            return False
        if self.target_location is None and (board_index is not None or store_index is not None):
            return False
        if self.target_location is not None:
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
        return True

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        pass
