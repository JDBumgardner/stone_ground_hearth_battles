import typing
from typing import Optional

from hearthstone.simulator.core.cards import CardLocation
from hearthstone.simulator.core.spell import Spell

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.events import BuyPhaseContext
    from hearthstone.simulator.core.player import BoardIndex, StoreIndex


class TripleRewardCard(Spell):
    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == self.tier)


class Prize(Spell):
    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == self.tier)


class RecruitmentMap(Spell):
    base_cost = 3

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == self.tier)


class GoldCoin(Spell):
    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.plus_coins(1)


class Banana(Spell):
    target_location = [CardLocation.BOARD, CardLocation.STORE]

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
        if store_index is not None:
            target = context.owner.store[store_index]
        target.attack += 1
        target.health += 1


class BigBanana(Spell):
    target_location = [CardLocation.BOARD, CardLocation.STORE]

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
        if store_index is not None:
            target = context.owner.store[store_index]
        target.attack += 2
        target.health += 2
