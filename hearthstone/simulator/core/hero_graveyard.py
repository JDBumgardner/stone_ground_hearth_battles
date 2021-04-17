from typing import Optional

from hearthstone.simulator.core.cards import one_minion_per_type, CardLocation
from hearthstone.simulator.core.events import BuyPhaseContext
from hearthstone.simulator.core.hero import Hero
from hearthstone.simulator.core.player import StoreIndex, BoardIndex


class QueenWagtoggle(Hero):
    base_power_cost = 1

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        for card in one_minion_per_type(context.owner.in_play, context.randomizer):
            card.attack += 2
            card.health += 1


class Galakrond(Hero):
    base_power_cost = 0
    power_target_location = [CardLocation.STORE]

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return bool(context.owner.store)

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        store_minion = context.owner.pop_store_card(store_index)
        context.owner.tavern.deck.return_cards(store_minion.dissolve())
        higher_tier_minions = [card for card in context.owner.tavern.deck.unique_cards() if
                               card.tier == min(store_minion.tier + 1, 6)]
        higher_tier_minion = context.randomizer.select_add_to_store(higher_tier_minions)
        context.owner.add_to_store(higher_tier_minion)


class CaptainHooktusk(Hero):
    base_power_cost = 1
    power_target_location = [CardLocation.BOARD]

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return context.owner.room_in_hand()

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        board_minion = context.owner.pop_board_card(board_index)
        context.owner.tavern.deck.return_cards(board_minion.dissolve())
        predicate = lambda card: (
                                     card.tier == board_minion.tier - 1 if board_minion.tier > 1 else card.tier == 1) and type(
            card) != type(board_minion)
        context.owner.draw_discover(predicate)
