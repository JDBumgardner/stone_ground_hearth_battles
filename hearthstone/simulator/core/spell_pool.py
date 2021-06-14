import collections
import sys
import typing
from inspect import getmembers, isclass
from typing import Optional

from hearthstone.simulator.core.cards import CardLocation
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.secrets import BaseSecret
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
    darkmoon_prize_tier = 1

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
        if store_index is not None:
            target = context.owner.store[store_index]
        target.attack += 2
        target.health += 2


class GachaGift(Spell):
    darkmoon_prize_tier = 1

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == 1)


class MightOfStormwind(Spell):
    darkmoon_prize_tier = 1

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        for card in context.owner.in_play:
            card.attack += 1
            card.health += 1


class PocketChange(Spell):
    darkmoon_prize_tier = 1

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        for _ in range(2):
            context.owner.gain_spell(GoldCoin())


class RockingAndRolling(Spell):
    darkmoon_prize_tier = 1

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.set_free_refreshes(3)


class NewRecruit(Spell):
    darkmoon_prize_tier = 1

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.new_recruit = True
        context.owner.extend_store(context.owner.tavern.deck.draw(context.owner, 1))


class TheGoodStuff(Spell):
    darkmoon_prize_tier = 1

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.the_good_stuff = True
        for card in context.owner.store:
            card.health += 1


class EvolvingTavern(Spell):
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        new_tiers = [min(6, card.tier + 1) for card in context.owner.store]
        context.owner.return_cards()
        for card_tier in new_tiers:
            higher_tier_minions = [card for card in context.owner.tavern.deck.unique_cards() if card.tier == card_tier]
            higher_tier_minion = context.randomizer.select_add_to_store(higher_tier_minions)
            context.owner.tavern.deck.remove_card(higher_tier_minion)
            context.owner.add_to_store(higher_tier_minion)


class GreatDeal(Spell):
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.tavern_cost_reduction += 2


class GruulRules(Spell):
    target_location = [CardLocation.BOARD, CardLocation.STORE]
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
        if store_index is not None:
            target = context.owner.store[store_index]
        target.gruul_rules = True


class OnTheHouse(Spell):
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == context.owner.tavern_tier)


class TheBouncer(Spell):
    target_location = [CardLocation.BOARD]
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
            target.attack += 5
            target.health += 5
            target.taunt = True


class TimeThief(Spell):
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        last_opp_warband_types = [type(card) for card in context.owner.last_opponent_warband]
        if last_opp_warband_types != [] and context.owner.room_in_hand():
            context.owner.draw_discover(lambda card: type(card) in last_opp_warband_types)


class TheUnlimitedCoin(Spell):
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.plus_coins(1)
        context.owner.the_unlimited_coins_played += 1


class BrannsBlessing(Spell):
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.battlecry_multiplier = 2


class AllThatGlitters(Spell):
    darkmoon_prize_tier = 2

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        random_store_minion = context.randomizer.select_from_store(context.owner.store)
        random_store_minion.golden_transformation([])


class Bananas(Spell):
    darkmoon_prize_tier = 3

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        num_bananas = context.owner.maximum_hand_size - context.owner.hand_size()
        for _ in range(num_bananas):
            if context.randomizer.select_random_number(1, 3) == 1:
                context.owner.gain_spell(BigBanana())
            else:
                context.owner.gain_spell(Banana())


class BuyTheHolyLight(Spell):
    target_location = [CardLocation.BOARD]
    darkmoon_prize_tier = 3

    def valid_target(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                     store_index: Optional['StoreIndex'] = None) -> bool:
        return not context.owner.in_play[board_index].divine_shield

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            context.owner.in_play[board_index].divine_shield = True


class ImStillJustARatInACage(Spell):
    target_location = [CardLocation.BOARD]
    darkmoon_prize_tier = 3

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
            target.attack += target.attack


class GainIceBlock(Spell):
    darkmoon_prize_tier = 3

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.secrets.append(BaseSecret.IceBlock())


class RepeatCustomer(Spell):
    target_location = [CardLocation.BOARD]
    darkmoon_prize_tier = 3

    def valid_target(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                     store_index: Optional['StoreIndex'] = None) -> bool:
        return not context.owner.in_play[board_index].golden

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.pop_board_card(board_index)
            target.attack += 2
            target.health += 2
            context.owner.gain_hand_card(target)


class TopShelf(Spell):
    darkmoon_prize_tier = 3

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.draw_discover(lambda card: card.tier == 6)


class TrainingSession(Spell):
    darkmoon_prize_tier = 3

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):

        # is there a better way to get around circular imports?
        from hearthstone.simulator.core.hero_pool import VALHALLA

        hero_pool = [hero_type() for hero_type in VALHALLA if (
                    hero_type.pool in context.owner.tavern.available_types or hero_type.pool == MONSTER_TYPES.ALL) and hero_type != type(
            context.owner.hero)]
        hero_choices = []
        for _ in range(3):
            random_hero = context.randomizer.select_hero(hero_pool)
            hero_choices.append(random_hero)
            hero_pool.remove(random_hero)
        context.owner.hero.discover_queue.append(hero_choices)
        context.owner.hero.player = context.owner


class GainArgentBraggart(Spell):
    darkmoon_prize_tier = 4

    def on_gain(self, context: 'BuyPhaseContext'):

        # is there a better way to get around circular imports?
        from hearthstone.simulator.core.card_pool import ArgentBraggart

        context.owner.remove_spell(self)
        context.owner.gain_hand_card(ArgentBraggart())


class FreshTab(Spell):
    darkmoon_prize_tier = 4

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.coins = context.owner.coin_income_rate


class FriendsAndFamilyDiscount(Spell):
    darkmoon_prize_tier = 4

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.minion_cost = context.owner.hero.minion_cost() - 1


class GiveADogABone(Spell):
    target_location = [CardLocation.BOARD]
    darkmoon_prize_tier = 4

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.in_play[board_index]
            target.divine_shield = True
            target.windfury = True
            target.attack += 10
            target.health += 10


class OpenBar(Spell):
    darkmoon_prize_tier = 4

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        context.owner.num_turn_start_free_refreshes = 5


class RaiseTheStakes(Spell):
    target_location = [CardLocation.BOARD]
    darkmoon_prize_tier = 4

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            target = context.owner.pop_board_card(board_index)
            target.golden_transformation([])
            context.owner.gain_hand_card(target)


class BigWinner(Spell):
    darkmoon_prize_tier = 4

    def on_play(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                store_index: Optional['StoreIndex'] = None):
        for i in range(1, 4):
            if context.owner.room_in_hand():
                prize_choies = DARKMOON_PRIZES[i]
                selected_prizes = []
                for _ in range(3):
                    spell_type = context.randomizer.select_spell(prize_choies)
                    selected_prizes.append(spell_type())
                    prize_choies.remove(spell_type)
                context.owner.hero.discover_queue.append(selected_prizes)


ALL_SPELLS = [member[1] for member in
              getmembers(sys.modules[__name__], lambda member: isclass(member) and member.__module__ == __name__)]

DARKMOON_PRIZES = collections.OrderedDict(
    [(tier, [spell for spell in ALL_SPELLS if spell.darkmoon_prize_tier == tier]) for tier in range(1, 5)])
