from typing import Optional

from hearthstone.simulator.core.cards import CardLocation
from hearthstone.simulator.core.events import BuyPhaseContext
from hearthstone.simulator.core.player import BoardIndex, StoreIndex
from hearthstone.simulator.core.spell import Spell


class TripleRewardCard(Spell):
    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def __repr__(self):
        return f"TripleRewards({self.level})"

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == self.level)


class Prize(Spell):
    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def __repr__(self):
        return f"Prize({self.level})"

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == self.level)


class RecruitmentMap(Spell):
    base_cost = 3

    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def __repr__(self):
        return f"RecruitmentMap({self.level})"

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == self.level)


class GoldCoin(Spell):
    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        context.owner.plus_coins(1)


class Banana(Spell):
    power_target_location = [CardLocation.BOARD, CardLocation.STORE]

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
        if store_index is not None:
            target = context.owner.store[store_index]
        target.attack += 1
        target.health += 1


class BigBanana(Spell):
    power_target_location = [CardLocation.BOARD, CardLocation.STORE]

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
        if store_index is not None:
            target = context.owner.store[store_index]
        target.attack += 2
        target.health += 2
