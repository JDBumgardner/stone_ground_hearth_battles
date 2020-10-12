from typing import Union, Tuple, Optional

from hearthstone.simulator.core.card_pool import Amalgam
from hearthstone.simulator.core.cards import one_minion_per_type, CardLocation
from hearthstone.simulator.core.events import BuyPhaseContext, CombatPhaseContext, EVENTS, CardEvent
from hearthstone.simulator.core.hero import Hero
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.player import BoardIndex, StoreIndex
from hearthstone.simulator.core.triple_reward_card import TripleRewardCard


class Pyramad(Hero):
    power_cost = 1

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        if context.owner.in_play:
            minion = context.randomizer.select_friendly_minion(context.owner.in_play)
            minion.health += 4


class LordJaraxxus(Hero):
    power_cost = 1
    pool = MONSTER_TYPES.DEMON

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        for minion in context.owner.in_play:
            if minion.check_type(MONSTER_TYPES.DEMON):
                minion.attack += 1
                minion.health += 1


class PatchWerk(Hero):
    def starting_health(self) -> int:
        return 50


class Nefarian(Hero):
    power_cost = 1

    # hero power is called nefarious fire

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event == EVENTS.COMBAT_START:
            if self.hero_power_used:
                for card in context.enemy_war_party.board:
                    card.take_damage(1, context.enemy_context())
                    card.resolve_death(context.enemy_context())


class Deathwing(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.COMBAT_START:
            all_minions = context.friendly_war_party.board + context.enemy_war_party.board
            for minion in all_minions:
                minion.attack += 2

        if event.event is EVENTS.SUMMON_COMBAT:
            event.card.attack += 2


class MillificentManastorm(Hero):
    pool = MONSTER_TYPES.MECH

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY:
            if event.card.check_type(MONSTER_TYPES.MECH):
                event.card.attack += 1
                event.card.health += 1


class YoggSaron(Hero):
    power_cost = 2

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        card = context.randomizer.select_from_store(context.owner.store)
        card.attack += 1
        card.health += 1
        context.owner.store.remove(card)
        context.owner.gain_hand_card(card)

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        if not context.owner.room_in_hand():
            return False

        if not context.owner.store:
            return False

        return True


class PatchesThePirate(Hero):
    power_cost = 4
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
        self.power_cost = 4

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
        if event.event is EVENTS.SELL and event.card.check_type(MONSTER_TYPES.MURLOC) and len(context.owner.store) < 7:
            murlocs = [card for card in context.owner.tavern.deck.unique_cards() if
                       card.check_type(MONSTER_TYPES.MURLOC) and card.tier <= context.owner.tavern_tier]
            card = context.randomizer.select_add_to_store(murlocs)
            context.owner.tavern.deck.remove_card(card)
            context.owner.store.append(card)


class KaelthasSunstrider(Hero):
    buy_counter = 0

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY:
            self.buy_counter += 1
            if self.buy_counter == 3:
                event.card.attack += 2
                event.card.health += 2
                self.buy_counter = 0


class LichBazhial(Hero):
    power_cost = 0

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return context.owner.room_in_hand()

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.take_damage(2)
        context.owner.gold_coins += 1


class SkycapnKragg(Hero):
    power_cost = 0

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.coins += context.owner.tavern.turn_count + 1
        self.can_use_power = False


class TheCurator(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START and context.owner.tavern.turn_count == 0:
            context.owner.gain_board_card(Amalgam())


class TheRatKing(Hero):
    current_type = None

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START:
            available_types = [monster_type for monster_type in MONSTER_TYPES.single_types() if
                               monster_type != self.current_type and monster_type in context.owner.tavern.available_types]
            self.current_type = context.randomizer.select_monster_type(available_types, context.owner.tavern.turn_count)

        if event.event is EVENTS.BUY and event.card.monster_type == self.current_type:
            event.card.attack += 1
            event.card.health += 2


class Ysera(Hero):
    pool = MONSTER_TYPES.DRAGON

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START and len(context.owner.store) < 7:
            dragons = [card for card in context.owner.tavern.deck.unique_cards() if
                       card.check_type(MONSTER_TYPES.DRAGON) and card.tier <= context.owner.tavern_tier]
            if dragons:
                card = context.randomizer.select_add_to_store(dragons)
                context.owner.tavern.deck.remove_card(card)
                context.owner.store.append(card)


class Bartendotron(Hero):
    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return (0, 4, 6, 7, 8, 9)


class MillhouseManastorm(Hero):
    def minion_cost(self) -> int:
        return 2

    def refresh_cost(self) -> int:
        return 2

    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return (0, 6, 8, 9, 10, 11)


class CaptainEudora(Hero):
    power_cost = 1
    digs_left = 5

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


class QueenWagtoggle(Hero):
    power_cost = 1

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        for card in one_minion_per_type(context.owner.in_play, context.randomizer):
            card.attack += 2


class ForestWardenOmu(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.TAVERN_UPGRADE:
            context.owner.coins += 2


class GeorgeTheFallen(Hero):
    power_cost = 3
    power_target_location = CardLocation.BOARD

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return not context.owner.in_play[board_index].divine_shield

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.in_play[board_index].divine_shield = True


class RenoJackson(Hero):
    power_cost = 0
    power_target_location = CardLocation.BOARD

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return not context.owner.in_play[board_index].golden

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.in_play[board_index].golden_transformation([])
        self.can_use_power = False


class JandiceBarov(Hero):
    power_cost = 0
    power_target_location = CardLocation.BOARD

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return not context.owner.in_play[board_index].golden and len(context.owner.store) > 0

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        board_minion = context.owner.pop_board_card(board_index)
        store_minion = context.randomizer.select_from_store(context.owner.store)
        context.owner.store.remove(store_minion)
        if store_minion.check_type(MONSTER_TYPES.ELEMENTAL):
            store_minion.attack += context.owner.nomi_bonus
            store_minion.health += context.owner.nomi_bonus
        context.owner.gain_board_card(store_minion)
        context.owner.store.append(board_minion)


class ArchVillianRafaam(Hero):  # TODO: tokens gained will enter the pool when sold
    power_cost = 1

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIES and event.card in context.enemy_war_party.board and self.hero_power_used:
            if len(context.enemy_war_party.dead_minions) == 1 and context.friendly_war_party.owner.room_in_hand():
                context.friendly_war_party.owner.gain_hand_card(type(event.card)())


class CaptainHooktusk(Hero):
    power_cost = 0
    power_target_location = CardLocation.BOARD

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return context.owner.room_in_hand()

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        board_minion = context.owner.pop_board_card(board_index)
        context.owner.tavern.deck.return_cards(board_minion.dissolve())
        predicate = lambda card: (card.tier < board_minion.tier if board_minion.tier > 1 else card.tier == 1) and type(
            card) != type(board_minion)
        lower_tier_minions = [card for card in context.owner.tavern.deck.unique_cards() if predicate(card)]
        random_minion = context.randomizer.select_gain_card(lower_tier_minions)
        context.owner.tavern.deck.remove_card(random_minion)
        context.owner.gain_hand_card(random_minion)


class Malygos(Hero):
    power_cost = 0
    power_target_location = CardLocation.BOARD

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        board_minion = context.owner.pop_board_card(board_index)
        context.owner.tavern.deck.return_cards(board_minion.dissolve())
        predicate = lambda card: card.tier == board_minion.tier and type(card) != type(board_minion)
        same_tier_minions = [card for card in context.owner.tavern.deck.unique_cards() if predicate(card)]
        random_minion = context.randomizer.select_gain_card(same_tier_minions)
        context.owner.tavern.deck.remove_card(random_minion)
        context.owner.gain_board_card(random_minion)


class AFKay(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START:
            if context.owner.tavern.turn_count in (0, 1):
                context.owner.coins = 0
            elif context.owner.tavern.turn_count == 2:
                for _ in range(2):
                    context.owner.triple_rewards.append(TripleRewardCard(3))


class EdwinVanCleef(Hero):
    power_cost = 1
    power_target_location = CardLocation.BOARD

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        bonus = len(context.owner.purchased_minions)
        context.owner.in_play[board_index].attack += bonus
        context.owner.in_play[board_index].health += bonus


class ArannaStarseeker(Hero):
    total_rerolls = 0

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.REFRESHED_STORE:
            self.total_rerolls += 1
            if self.total_rerolls >= 4:
                context.owner.store.extend(context.owner.tavern.deck.draw(context.owner, 7 - len(context.owner.store)))


class DinotamerBrann(Hero):
    power_cost = 1

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        context.owner.return_cards()
        number_of_cards = 3 + context.owner.tavern_tier // 2 - len(context.owner.store)
        predicate = lambda card: card.base_battlecry
        context.owner.store.extend(
            [context.owner.tavern.deck.draw_with_predicate(context.owner, predicate) for _ in range(number_of_cards)])


class Alexstrasza(Hero):
    pool = MONSTER_TYPES.DRAGON

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.TAVERN_UPGRADE and context.owner.tavern_tier == 5:
            for _ in range(2):
                if context.owner.room_in_hand():
                    context.owner.draw_discover(lambda card: card.check_type(MONSTER_TYPES.DRAGON))


class KingMukla(Hero):
    power_cost = 1

    def hero_power_valid_impl(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return context.owner.room_in_hand()

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        for _ in range(2):
            if context.owner.room_in_hand():
                context.owner.bananas += 1

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END and self.hero_power_used:
            for player in context.owner.tavern.players.values():
                if player != context.owner and player.room_in_hand():
                    player.bananas += 1


class EliseStarseeker(Hero):
    power_cost = 3
    multiple_power_uses_per_turn = True
    recruitment_maps = []

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


class AlAkir(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.COMBAT_START and len(context.friendly_war_party.board) >= 1:
            leftmost_minion = context.friendly_war_party.board[0]
            leftmost_minion.windfury = True
            leftmost_minion.divine_shield = True
            leftmost_minion.taunt = True


class Chenvaala(Hero):
    play_counter = 0
    pool = MONSTER_TYPES.ELEMENTAL

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SUMMON_BUY and event.card.check_type(MONSTER_TYPES.ELEMENTAL):
            self.play_counter += 1
            if self.play_counter == 3:
                context.owner.tavern_upgrade_cost -= 2
                self.play_counter = 0


class RagnarosTheFirelord(Hero):
    minions_killed = 0
    sulfuras = False

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIES and event.card in context.enemy_war_party.board:
            self.minions_killed += 1
            if self.minions_killed == 20:
                self.sulfuras = True
        if event.event is EVENTS.BUY_END and self.sulfuras and len(context.owner.in_play) >= 1:
            for i in [0, -1]:  # TODO: not 100% sure this is correct with only 1 minion in play
                context.owner.in_play[i].attack += 4
                context.owner.in_play[i].health += 4


class Rakanishu(Hero):
    power_cost = 2

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return bool(context.owner.in_play)

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        random_minion = context.randomizer.select_friendly_minion(context.owner.in_play)
        random_minion.attack += context.owner.tavern_tier
        random_minion.health += context.owner.tavern_tier


class MrBigglesworth(Hero):  # TODO: tokens discovered will enter the pool when sold
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.PLAYER_DEAD and bool(event.player.in_play):
            discovered_cards = []
            for _ in range(3):
                if event.player.in_play:
                    enemy_minion = context.randomizer.select_enemy_minion(event.player.in_play)
                    event.player.remove_board_card(enemy_minion)  # TODO: need to keep these three minions on the board
                    discovered_cards.append(enemy_minion)
            context.owner.discover_queue.append(discovered_cards)


class Nozdormu(Hero):
    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START:
            context.owner.free_refreshes = max(1, context.owner.free_refreshes)
