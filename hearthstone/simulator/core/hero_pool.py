import copy
import logging
import sys
from inspect import getmembers, isclass
from typing import Union, Tuple, Optional

from hearthstone.simulator.core import combat, events, hero
from hearthstone.simulator.core.card_pool import Amalgam, EmperorCobra, Snake, FishOfNZoth
from hearthstone.simulator.core.cards import one_minion_per_type, CardLocation
from hearthstone.simulator.core.events import BuyPhaseContext, CombatPhaseContext, EVENTS, CardEvent
from hearthstone.simulator.core.hero import Hero
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.player import BoardIndex, StoreIndex, DiscoverIndex, HeroChoiceIndex
from hearthstone.simulator.core.secrets import SECRETS
from hearthstone.simulator.core.triple_reward_card import TripleRewardCard


class Pyramad(Hero):
    base_power_cost = 1

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        if context.owner.in_play:
            minion = context.randomizer.select_friendly_minion(context.owner.in_play)
            minion.health += 4


class LordJaraxxus(Hero):
    base_power_cost = 1
    pool = MONSTER_TYPES.DEMON

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        for minion in context.owner.in_play:
            if minion.check_type(MONSTER_TYPES.DEMON):
                minion.attack += 1
                minion.health += 1


class PatchWerk(Hero):
    def starting_health(self) -> int:
        return 55


class Deathwing(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.COMBAT_PREPHASE:
            all_minions = context.friendly_war_party.board + context.enemy_war_party.board
            for minion in all_minions:
                minion.attack += 2

        if event.event is EVENTS.SUMMON_COMBAT:
            event.card.attack += 2


class MillificentManastorm(Hero):
    pool = MONSTER_TYPES.MECH

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.ADD_TO_STORE:
            if event.card.check_type(MONSTER_TYPES.MECH):
                event.card.attack += 1
                event.card.health += 1


class YoggSaron(Hero):
    base_power_cost = 2

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        card = context.randomizer.select_from_store(context.owner.store)
        card.attack += 1
        card.health += 1
        context.owner.remove_store_card(card)
        context.owner.gain_hand_card(card)

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        if not context.owner.room_in_hand():
            return False
        if not context.owner.store:
            return False
        return True


class PatchesThePirate(Hero):
    base_power_cost = 3
    pool = MONSTER_TYPES.PIRATE

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY and event.card.check_type(MONSTER_TYPES.PIRATE):
            self.power_cost = max(0, self.power_cost - 1)

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        pirates = [card for card in context.owner.tavern.deck.unique_cards() if
                   card.check_type(MONSTER_TYPES.PIRATE) and card.tier <= context.owner.tavern_tier]
        if pirates:
            card = context.randomizer.select_gain_card(pirates)
            context.owner.tavern.deck.remove_card(card)
            context.owner.gain_hand_card(card)
        self.power_cost = 3

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return context.owner.room_in_hand()


class DancinDeryl(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SELL and context.owner.store:
            for _ in range(2):
                card = context.randomizer.select_from_store(context.owner.store)
                card.attack += 1
                card.health += 1


class FungalmancerFlurgl(Hero):
    pool = MONSTER_TYPES.MURLOC

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SELL and event.card.check_type(MONSTER_TYPES.MURLOC) and context.owner.store_size() < context.owner.maximum_store_size:
            murlocs = [card for card in context.owner.tavern.deck.unique_cards() if
                       card.check_type(MONSTER_TYPES.MURLOC) and card.tier <= context.owner.tavern_tier]
            card = context.randomizer.select_add_to_store(murlocs)
            context.owner.tavern.deck.remove_card(card)
            context.owner.add_to_store(card)


class KaelthasSunstrider(Hero):
    buy_counter = 0

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY:
            self.buy_counter += 1
            if self.buy_counter == 3:
                event.card.attack += 2
                event.card.health += 2
                self.buy_counter = 0

    def hero_info(self) -> Optional[str]:
        return f'{3-self.buy_counter} buys left until hero power bonus'


class LichBazhial(Hero):
    base_power_cost = 0

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return context.owner.room_in_hand()

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.take_damage(2)
        context.owner.gold_coins += 1


class SkycapnKragg(Hero):
    base_power_cost = 0

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.coins += context.owner.tavern.turn_count + 1
        self.can_use_power = False


class TheCurator(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START and context.owner.tavern.turn_count == 0:
            context.owner.gain_board_card(Amalgam())


class TheRatKing(Hero):

    def __init__(self):
        super().__init__()
        self.current_type = None

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START:
            available_types = [monster_type for monster_type in MONSTER_TYPES.single_types() if
                               monster_type != self.current_type and monster_type in context.owner.tavern.available_types]
            self.current_type = context.randomizer.select_monster_type(available_types, context.owner.tavern.turn_count)

        if event.event is EVENTS.BUY and event.card.monster_type == self.current_type:
            event.card.attack += 2
            event.card.health += 2

    def hero_info(self) -> Optional[str]:
        return f'current type bonus: {self.current_type}'


class Ysera(Hero):
    pool = MONSTER_TYPES.DRAGON

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.REFRESHED_STORE or event.event is EVENTS.BUY_START and context.owner.store_size() < context.owner.maximum_store_size:
            dragons = [card for card in context.owner.tavern.deck.unique_cards() if
                       card.check_type(MONSTER_TYPES.DRAGON) and card.tier <= context.owner.tavern_tier]
            if dragons:
                card = context.randomizer.select_add_to_store(dragons)
                context.owner.tavern.deck.remove_card(card)
                context.owner.add_to_store(card)


class MillhouseManastorm(Hero):
    def minion_cost(self) -> int:
        return 2

    def refresh_cost(self) -> int:
        return 2

    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return 0, 6, 8, 9, 10, 11


class CaptainEudora(Hero):
    base_power_cost = 1

    def __init__(self):
        super().__init__()
        self.digs_left = 5

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return self.digs_left != 1 or context.owner.room_in_hand()

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        self.digs_left -= 1
        if self.digs_left == 0:
            diggable_minions = [card for card in context.owner.tavern.deck.unique_cards() if
                                card.tier <= context.owner.tavern_tier]
            random_minion = context.randomizer.select_gain_card(diggable_minions)
            context.owner.tavern.deck.remove_card(random_minion)
            random_minion.golden_transformation([])
            context.owner.gain_hand_card(random_minion)
            self.digs_left = 5

    def hero_info(self) -> Optional[str]:
        return f'{self.digs_left} digs left'


class ForestWardenOmu(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.TAVERN_UPGRADE:
            context.owner.coins += 2


class GeorgeTheFallen(Hero):
    base_power_cost = 2
    power_target_location = [CardLocation.BOARD]

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return not context.owner.in_play[board_index].divine_shield

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.in_play[board_index].divine_shield = True


class RenoJackson(Hero):
    base_power_cost = 0
    power_target_location = [CardLocation.BOARD]

    def __init__(self):
        super().__init__()
        self.target = None

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return not context.owner.in_play[board_index].golden

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        target = context.owner.in_play[board_index]
        target.golden_transformation([])
        self.target = target
        self.can_use_power = False

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SELL and event.card == self.target:
            event.card.golden = False  # only add 1 copy to pool when sold


class JandiceBarov(Hero):
    base_power_cost = 0
    power_target_location = [CardLocation.BOARD]

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return not context.owner.in_play[board_index].golden and context.owner.store_size() > 0

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        board_minion = context.owner.pop_board_card(board_index)
        store_minion = context.randomizer.select_from_store(context.owner.store)
        context.owner.remove_store_card(store_minion)
        context.owner.gain_board_card(store_minion)
        context.owner.add_to_store(board_minion)


class ArchVillianRafaam(Hero):
    base_power_cost = 1

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIES and event.card in context.enemy_war_party.board and self.hero_power_used:
            if len(context.enemy_war_party.dead_minions) == 1 and context.friendly_war_party.owner.room_in_hand():
                card_copy = type(event.card)()
                card_copy.token = False
                context.friendly_war_party.owner.gain_hand_card(card_copy)


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
        predicate = lambda card: (card.tier == board_minion.tier-1 if board_minion.tier > 1 else card.tier == 1) and type(
            card) != type(board_minion)
        context.owner.draw_discover(predicate)


class Malygos(Hero):
    base_power_cost = 0
    power_target_location = [CardLocation.BOARD, CardLocation.STORE]  # TODO: are there other hero powers with multiple target locations?

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        if board_index is not None:
            minion = context.owner.pop_board_card(board_index)
        elif store_index is not None:
            minion = context.owner.pop_store_card(store_index)
        context.owner.tavern.deck.return_cards(minion.dissolve())
        predicate = lambda card: card.tier == minion.tier and type(card) != type(minion)
        same_tier_minions = [card for card in context.owner.tavern.deck.unique_cards() if predicate(card)]
        random_minion = context.randomizer.select_gain_card(same_tier_minions)
        context.owner.tavern.deck.remove_card(random_minion)
        if board_index is not None:
            context.owner.gain_board_card(random_minion)
        elif store_index is not None:
            context.owner.add_to_store(random_minion)


class AFKay(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START:
            if context.owner.tavern.turn_count in (0, 1):
                context.owner.coins = 0
            elif context.owner.tavern.turn_count == 2:
                for _ in range(2):
                    context.owner.triple_rewards.append(TripleRewardCard(3))


class EdwinVanCleef(Hero):
    base_power_cost = 1
    power_target_location = [CardLocation.BOARD]  # TODO: can this target minions in the store?

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        bonus = len(context.owner.purchased_minions)
        context.owner.in_play[board_index].attack += bonus
        context.owner.in_play[board_index].health += bonus


class ArannaStarseeker(Hero):
    def __init__(self):
        super().__init__()
        self.total_rerolls = 0

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.REFRESHED_STORE or event.event is EVENTS.BUY_START:
            self.total_rerolls += 1 if event.event is EVENTS.REFRESHED_STORE else 0
            if self.total_rerolls >= 5:
                context.owner.extend_store(context.owner.tavern.deck.draw(context.owner, 7 - context.owner.store_size()))

    def hero_info(self) -> Optional[str]:
        return f'{max(5-self.total_rerolls, 0)} refreshes left'


class DinotamerBrann(Hero):
    base_power_cost = 1

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.return_cards()
        predicate = lambda card: card.base_battlecry
        context.owner.extend_store(
            [context.owner.tavern.deck.draw_with_predicate(context.owner, predicate) for _ in range(context.owner.refresh_size())])


class Alexstrasza(Hero):
    pool = MONSTER_TYPES.DRAGON

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.TAVERN_UPGRADE and context.owner.tavern_tier == 5:
            for _ in range(2):
                if context.owner.room_in_hand():
                    context.owner.draw_discover(lambda card: card.check_type(MONSTER_TYPES.DRAGON))


class KingMukla(Hero):
    base_power_cost = 1

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return context.owner.room_in_hand()

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        for _ in range(2):
            if context.owner.room_in_hand():
                context.owner.bananas += 1
                if context.randomizer.select_random_number(1, 5) == 5:
                    context.owner.big_bananas += 1

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END and self.hero_power_used:
            for player in context.owner.tavern.players.values():
                if player != context.owner and player.room_in_hand():
                    player.bananas += 1


class EliseStarseeker(Hero):
    base_power_cost = 2
    multiple_power_uses_per_turn = True

    def __init__(self):
        super().__init__()
        self.recruitment_maps = []

    def occupied_hand_slots(self) -> int:
        return len(self.recruitment_maps)

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.TAVERN_UPGRADE and context.owner.room_in_hand():
            self.recruitment_maps.append(context.owner.tavern_tier)

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return bool(self.recruitment_maps)

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        discover_tier = self.recruitment_maps.pop()
        context.owner.draw_discover(lambda card: card.tier == discover_tier)

    def hero_info(self) -> Optional[str]:
        return f'current recruitment maps: {["tier " + str(map) for map in self.recruitment_maps]}'


class AlAkir(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.COMBAT_START and len(context.friendly_war_party.board) >= 1:
            leftmost_minion = context.friendly_war_party.board[0]
            leftmost_minion.windfury = True
            leftmost_minion.divine_shield = True
            leftmost_minion.taunt = True


class Chenvaala(Hero):
    pool = MONSTER_TYPES.ELEMENTAL

    def __init__(self):
        super().__init__()
        self.play_counter = 0

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SUMMON_BUY and event.card.check_type(MONSTER_TYPES.ELEMENTAL):
            self.play_counter += 1
            if self.play_counter == 3:
                context.owner.tavern_upgrade_cost -= 3
                self.play_counter = 0

    def hero_info(self) -> Optional[str]:
        return f'{3 - self.play_counter} elemental plays left until tavern discount'


class RagnarosTheFirelord(Hero):

    def __init__(self):
        super().__init__()
        self.minions_killed = 0
        self.sulfuras = False

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIES and event.card in context.enemy_war_party.board:
            self.minions_killed += 1
            if self.minions_killed == 25:
                self.sulfuras = True
        if event.event is EVENTS.BUY_END and self.sulfuras and len(context.owner.in_play) >= 1:
            for i in [0, -1]:
                context.owner.in_play[i].attack += 3
                context.owner.in_play[i].health += 3

    def hero_info(self) -> Optional[str]:
        return f'{max(25-self.minions_killed, 0)} minions left'


class Rakanishu(Hero):
    base_power_cost = 2
    power_target_location = [CardLocation.BOARD]

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        board_minion = context.owner.in_play[board_index]
        board_minion.attack += context.owner.tavern_tier
        board_minion.health += context.owner.tavern_tier


class MrBigglesworth(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.PLAYER_DEAD and bool(event.player.in_play):
            discovered_cards = []
            board = [copy.deepcopy(card) for card in event.player.in_play]
            for _ in range(3):
                if board:
                    enemy_minion = context.randomizer.select_enemy_minion(board)
                    board.remove(enemy_minion)
                    enemy_minion.token = True
                    discovered_cards.append(enemy_minion)
            context.owner.discover_queue.append(discovered_cards)


class Nozdormu(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START:
            context.owner.free_refreshes = max(1, context.owner.free_refreshes)


class Sindragosa(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END:
            for card in context.owner.store:
                if card.frozen:
                    card.attack += 2
                    card.health += 1


class InfiniteToki(Hero):
    base_power_cost = 1

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.return_cards()
        number_of_cards = min(context.owner.refresh_size(), context.owner.maximum_store_size - context.owner.store_size())
        context.owner.extend_store(context.owner.tavern.deck.draw(context.owner, number_of_cards - 1))
        if context.owner.maximum_store_size > context.owner.store_size():
            higher_tier_minions = [card for card in context.owner.tavern.deck.unique_cards() if
                                   card.tier == min(context.owner.tavern_tier + 1, context.owner.max_tier())]
            higher_tier_minion = context.randomizer.select_add_to_store(higher_tier_minions)
            context.owner.tavern.deck.remove_card(higher_tier_minion)
            context.owner.add_to_store(higher_tier_minion)


class TheLichKing(Hero):
    base_power_cost = 0
    power_target_location = [CardLocation.BOARD]

    def __init__(self):
        super().__init__()
        self.target = None
        self.target_index = None

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return not context.owner.in_play[board_index].reborn

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        self.target = context.owner.in_play[board_index]

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SELL and event.card == self.target:
            self.target = None
        if event.event is EVENTS.BUY_END and self.target is not None:
            if self.target in context.owner.in_play:
                self.target_index = context.owner.in_play.index(self.target)
            self.target = None
        if event.event is EVENTS.COMBAT_PREPHASE and self.target_index is not None:
            context.friendly_war_party.board[self.target_index].reborn = True
            self.target_index = None


class TessGreymane(Hero):
    base_power_cost = 1

    def __init__(self):
        super().__init__()
        self.refreshed_cards = []

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        if context.owner.last_opponent_warband:
            context.owner.return_cards()
            for card in context.owner.last_opponent_warband:
                card_copy = type(card)()
                card_copy.token = True
                context.owner.add_to_store(card_copy)
                if not card.base_token:
                    self.refreshed_cards.append(card_copy)
            context.owner.extend_store(context.owner.tavern.deck.draw(context.owner, context.owner.refresh_size()))
        else:
            context.owner.draw()

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY and event.card in self.refreshed_cards:
            event.card.token = False
            self.refreshed_cards.remove(event.card)


class Shudderwock(Hero):
    base_power_cost = 1

    def __init__(self):
        super().__init__()
        self.battlecries_counted = 0

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        self.battlecries_counted = 0

    def battlecry_multiplier(self) -> int:
        if self.hero_power_used and self.battlecries_counted == 1:
            return 2
        return 1

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SUMMON_BUY and event.card.battlecry and self.hero_power_used:
            self.battlecries_counted += 1


class TheGreatAkazamzarak(Hero):
    base_power_cost = 1

    def __init__(self):
        super().__init__()
        self.secrets = []

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return len(self.secrets) <= 5

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        available_secrets = SECRETS.remaining_secrets(self)
        for _ in range(3):
            secret = context.randomizer.select_secret(available_secrets)
            available_secrets.remove(secret)
            self.discover_choices.append(secret)

    def select_discover(self, discover_index: 'DiscoverIndex'):
        self.secrets.append(self.discover_choices[discover_index])
        self.discover_choices = []

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START and SECRETS.COMPETETIVE_SPIRIT in self.secrets:
            logging.debug(f'{SECRETS.COMPETETIVE_SPIRIT} triggers')
            self.secrets.remove(SECRETS.COMPETETIVE_SPIRIT)
            for card in context.owner.in_play:
                card.attack += 1
                card.health += 1
        if event.event is EVENTS.BUY_END:
            self.give_immunity = False
        if event.event is EVENTS.END_COMBAT and context.owner.health <= 0 and SECRETS.ICE_BLOCK in self.secrets:
            logging.debug(f'{SECRETS.ICE_BLOCK} triggers')
            self.secrets.remove(SECRETS.ICE_BLOCK)
            context.owner.health += event.damage_taken
            self.give_immunity = True
        if event.event is EVENTS.IS_ATTACKED and event.card in context.friendly_war_party.board:
            if context.friendly_war_party.room_on_board():
                if SECRETS.SPLITTING_IMAGE in self.secrets and context.friendly_war_party.room_on_board():
                    logging.debug(f'{SECRETS.SPLITTING_IMAGE} triggers')
                    self.secrets.remove(SECRETS.SPLITTING_IMAGE)
                    summon_index = context.friendly_war_party.get_index(event.card)
                    for i in range(context.summon_minion_multiplier()):
                        context.friendly_war_party.summon_in_combat(copy.deepcopy(event.card), context, summon_index+1+i)
                if SECRETS.VENOMSTRIKE_TRAP in self.secrets and context.friendly_war_party.room_on_board():
                    logging.debug(f'{SECRETS.VENOMSTRIKE_TRAP} triggers')
                    self.secrets.remove(SECRETS.VENOMSTRIKE_TRAP)
                    for _ in range(context.summon_minion_multiplier()):
                        cobra = EmperorCobra()
                        context.friendly_war_party.summon_in_combat(cobra, context)
                if SECRETS.SNAKE_TRAP in self.secrets and context.friendly_war_party.room_on_board():
                    self.secrets.remove(SECRETS.SNAKE_TRAP)
                    logging.debug(f'{SECRETS.SNAKE_TRAP} triggers')
                    for _ in range(3 * context.summon_minion_multiplier()):
                        snake = Snake()
                        context.friendly_war_party.summon_in_combat(snake, context)
            if SECRETS.AUTODEFENSE_MATRIX in self.secrets and not event.card.divine_shield:
                logging.debug(f'{SECRETS.AUTODEFENSE_MATRIX} triggers')
                self.secrets.remove(SECRETS.AUTODEFENSE_MATRIX)
                event.card.divine_shield = True

        if event.event is EVENTS.DIES and event.card in context.friendly_war_party.board:
            if SECRETS.REDEMPTION in self.secrets:
                logging.debug(f'{SECRETS.REDEMPTION} triggers')
                self.secrets.remove(SECRETS.REDEMPTION)
                summon_index = context.friendly_war_party.get_index(event.card)
                for i in range(context.summon_minion_multiplier()):
                    new_copy = event.card.unbuffed_copy()
                    new_copy.health = 1
                    context.friendly_war_party.summon_in_combat(new_copy, context, summon_index+1+i)

            if SECRETS.AVENGE in self.secrets and context.friendly_war_party.live_minions():
                logging.debug(f'{SECRETS.AVENGE} triggers')
                self.secrets.remove(SECRETS.AVENGE)
                random_friend = context.randomizer.select_friendly_minion(context.friendly_war_party.live_minions())
                random_friend.attack += 3
                random_friend.health += 2

    def hero_info(self) -> Optional[str]:
        return f'active secrets: {[secret.name for secret in self.secrets]}'


class IllidanStormrage(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.COMBAT_START:
            if context.friendly_war_party.board:
                for i in {0, len(context.friendly_war_party.board) - 1}:
                    attacking_war_party = context.friendly_war_party
                    defending_war_party = context.enemy_war_party
                    attacker = context.friendly_war_party.board[i]
                    num_attacks = attacker.num_attacks() if attacker else 1
                    for _ in range(num_attacks):
                        defender = defending_war_party.get_attack_target(context.randomizer, attacker)
                        if defender is None or attacker.dead:
                            break
                        logging.debug(f'{attacking_war_party.owner.name} is attacking {defending_war_party.owner.name} from Illidan Stormrage\'s effect')
                        combat.start_attack(attacker, defender, attacking_war_party, defending_war_party, context.randomizer)


class ZephrysTheGreat(Hero):
    base_power_cost = 3

    def __init__(self):
        super().__init__()
        self.wishes_left = 3

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return self.wishes_left > 0

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        pairs = [minion for minion in context.owner.in_play if
                 not minion.golden and len([card for card in context.owner.in_play if type(card) == type(minion)]) == 2]
        if pairs:
            pair = context.randomizer.select_friendly_minion(pairs)
            context.owner.gain_hand_card(type(pair)())  # TODO: How does this interact with the minion pool?
        self.wishes_left -= 1

    def hero_info(self) -> Optional[str]:
        return f'{self.wishes_left} wishes left'


class SilasDarkmoon(Hero):
    def __init__(self):
        super().__init__()
        self.tickets_purchased = 0

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.ADD_TO_STORE:
            event.card.ticket = bool(context.randomizer.select_random_number(0, 1))  # TODO: what should the odds be?
        if event.event is EVENTS.BUY:
            if event.card.ticket:
                self.tickets_purchased += 1
                event.card.ticket = False
                if self.tickets_purchased == 3:
                    self.tickets_purchased = 0
                    context.owner.triple_rewards.append(TripleRewardCard(context.owner.tavern_tier))

    def hero_info(self) -> Optional[str]:
        return f'{3-self.tickets_purchased} ticket buys left until discover reward'


class SirFinleyMrrgglton(Hero):
    def __init__(self):
        super().__init__()
        self.player = None

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START and context.owner.tavern.turn_count == 0:
            hero_pool = [hero_type() for hero_type in VALHALLA if (hero_type.pool in context.owner.tavern.available_types or hero_type.pool == MONSTER_TYPES.ALL) and hero_type != type(self)]
            hero_choices = []
            for _ in range(3):
                random_hero = context.randomizer.select_hero(hero_pool)
                hero_choices.append(random_hero)
                hero_pool.remove(random_hero)
            self.discover_choices.extend(hero_choices)
            self.player = context.owner

    def select_discover(self, discover_index: 'DiscoverIndex'):
        self.player.hero_options = self.discover_choices[:]
        self.player.choose_hero(HeroChoiceIndex(discover_index))
        self.player.hero.handle_event(events.BuyStartEvent(), BuyPhaseContext(self.player, self.player.tavern.randomizer))
        self.discover_choices = []


class LordBarov(Hero):
    base_power_cost = 1

    def __init__(self):
        super().__init__()
        self.winning_pick = None

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        if len(context.owner.tavern.current_player_pairings) > 1:
            available_pairings = [(player1, player2) for player1, player2 in
                                  context.owner.tavern.current_player_pairings if
                                  player1 != context.owner and player2 != context.owner]
            pairing = context.randomizer.select_combat_matchup(available_pairings)
        else:
            pairing = context.owner.tavern.current_player_pairings[0]
        self.discover_choices.extend(list(pairing))

    def select_discover(self, discover_index: 'DiscoverIndex'):
        self.winning_pick = self.discover_choices[discover_index]
        self.discover_choices = []

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.RESULTS_BROADCAST and self.winning_pick is not None:
            if event.winner == self.winning_pick:
                context.owner.gold_coins += 3
                self.winning_pick = None
            elif event.tie:
                context.owner.gold_coins += 1
                self.winning_pick = None
            elif event.loser == self.winning_pick:
                self.winning_pick = None


class MaievShadowsong(Hero):
    base_power_cost = 1
    power_target_location = [CardLocation.STORE]

    def __init__(self):
        super().__init__()
        self.dormant_minions = dict()

    def occupied_store_slots(self) -> int:
        return len(self.dormant_minions)

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        store_minion = context.owner.pop_store_card(store_index)
        self.dormant_minions[store_minion] = 2

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START:
            for card in list(self.dormant_minions.keys()):
                self.dormant_minions[card] -= 1
                if self.dormant_minions[card] == 0:
                    context.owner.gain_hand_card(card)
                    card.attack += 1
                    del self.dormant_minions[card]

    def hero_info(self) -> Optional[str]:
        return f"dormant minions: {self.dormant_minions}"


class CThun(Hero):
    base_power_cost = 2

    def __init__(self):
        super().__init__()
        self.power_uses = 0

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        self.power_uses += 1

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END and context.owner.in_play:
            for _ in range(self.power_uses):
                random_minion = context.randomizer.select_friendly_minion(context.owner.in_play)
                random_minion.attack += 1
                random_minion.health += 1

    def hero_info(self) -> Optional[str]:
        return f'hero power repeats {self.power_uses} times'


class YShaarj(Hero):
    base_power_cost = 2

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.COMBAT_START and self.hero_power_used and context.friendly_war_party.room_on_board():  # TODO: order of this vs other start of combat effects?
            same_tier_options = [card for card in context.friendly_war_party.owner.tavern.deck.unique_cards() if
                                 card.tier == context.friendly_war_party.owner.tavern_tier]
            random_minion = context.randomizer.select_gain_card(same_tier_options)
            context.friendly_war_party.summon_in_combat(type(random_minion)(), context)
            context.friendly_war_party.owner.gain_board_card(random_minion)


class NZoth(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START and context.owner.tavern.turn_count == 0:
            context.owner.gain_board_card(FishOfNZoth())


class Greybough(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board:
            event.card.attack += 1
            event.card.health += 2
            event.card.taunt = True


# TODO: add Tickatus... and darkmoon prizes (ugh)


VALHALLA = [member[1] for member in getmembers(sys.modules[__name__], lambda member: isclass(member) and member.__module__ == __name__)]
