from typing import Union, Tuple

from hearthstone.card_pool import Amalgam
from hearthstone.cards import CardEvent, PrintingPress, one_minion_per_type
from hearthstone.events import BuyPhaseContext, CombatPhaseContext, EVENTS
from hearthstone.hero import Hero
from hearthstone.monster_types import MONSTER_TYPES


class Pyramad(Hero):
    power_cost = 1

    def hero_power_impl(self, context: BuyPhaseContext):
        if context.owner.in_play:
            minion = context.randomizer.select_friendly_minion(context.owner.in_play)
            minion.health += 4


class LordJaraxxus(Hero):
    power_cost = 1

    def hero_power_impl(self, context: BuyPhaseContext):
        for minion in context.owner.in_play:
            if minion.check_type(MONSTER_TYPES.DEMON):
                minion.attack += 1
                minion.health += 1


class PatchWerk(Hero):
    def starting_health(self) -> int:
        return 50

    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False


class Nefarian(Hero):
    power_cost = 1

    # hero power is called nefarious fire

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == EVENTS.COMBAT_START:
            if self.hero_power_used:
                for card in context.enemy_war_party.board:
                    card.take_damage(1, context)
                    card.resolve_death(context)


class Deathwing(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.COMBAT_START:
            all_minions = context.friendly_war_party.board + context.enemy_war_party.board
            for minion in all_minions:
                minion.attack += 2

        if event.event is EVENTS.SUMMON_COMBAT:
            event.card.attack += 2


class MillificentManastorm(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.BUY:
            if event.card.check_type(MONSTER_TYPES.MECH):
                event.card.attack += 1
                event.card.health += 1


class YoggSaron(Hero):
    power_cost = 2

    def hero_power_impl(self, context: BuyPhaseContext):
        card = context.randomizer.select_from_store(context.owner.store)
        card.attack += 1
        card.health += 1
        context.owner.store.remove(card)
        context.owner.gain_card(card)

    def hero_power_valid_impl(self, context: BuyPhaseContext):
        if not context.owner.room_in_hand():
            return False

        if not context.owner.store:
            return False

        return True


class PatchesThePirate(Hero):  # TODO: does this pull from the deck or does it add to the available pool?
    power_cost = 4

    def handle_event(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.BUY and event.card.check_type(MONSTER_TYPES.PIRATE):
            self.power_cost = max(0, self.power_cost - 1)

    def hero_power_impl(self, context: BuyPhaseContext):
        pirates = [card for card in context.owner.tavern.deck.all_cards() if
                   card.check_type(MONSTER_TYPES.PIRATE) and card.tier <= context.owner.tavern_tier]

        card = context.randomizer.select_gain_card(pirates)
        context.owner.tavern.deck.remove_card(card)
        context.owner.gain_card(card)
        self.power_cost = 4

    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return context.owner.room_in_hand()


class DancinDeryl(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.SELL and context.owner.store:
            for _ in range(2):
                card = context.randomizer.select_from_store(context.owner.store)
                card.attack += 1
                card.health += 1


class FungalmancerFlurgl(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.SELL and event.card.check_type(MONSTER_TYPES.MURLOC):
            murlocs = [card for card in context.owner.tavern.deck.all_cards() if
                       card.check_type(MONSTER_TYPES.MURLOC) and card.tier <= context.owner.tavern_tier]
            card = context.randomizer.select_add_to_store(murlocs)
            context.owner.tavern.deck.remove_card(card)
            context.owner.store.append(card)


class KaelthasSunstrider(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.BUY:
            self.buy_counter += 1
            if self.buy_counter == 3:
                event.card.attack += 2
                event.card.health += 2
                self.buy_counter = 0


class LichBazhial(Hero):
    power_cost = 0

    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return context.owner.room_in_hand()

    def hero_power_impl(self, context: BuyPhaseContext):
        context.owner.take_damage(2)
        context.owner.gold_coins += 1


class SkycapnKragg(Hero):
    power_cost = 0

    def hero_power_impl(self, context: BuyPhaseContext):
        if self.can_use_power:
            context.owner.coins += context.owner.tavern.turn_count + 1
            self.can_use_power = False


class TheCurator(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.BUY_START and context.owner.tavern.turn_count == 0:
            context.owner.in_play.append(Amalgam())


class TheRatKing(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.BUY_START:
            available_types = [monster_type for monster_type in MONSTER_TYPES.single_types() if monster_type != self.current_type]
            self.current_type = context.randomizer.select_monster_type(available_types, context.owner.tavern.turn_count)

        if event.event is EVENTS.BUY and event.card.monster_type == self.current_type:
            event.card.attack += 1
            event.card.health += 2


class Ysera(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.BUY_START and len(context.owner.store) < 7:
            dragons = [card for card in context.owner.tavern.deck.all_cards() if
                       card.check_type(MONSTER_TYPES.DRAGON) and card.tier <= context.owner.tavern_tier]
            card = context.randomizer.select_add_to_store(dragons)
            context.owner.tavern.deck.remove_card(card)
            context.owner.store.append(card)


class Bartendotron(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return (0, 4, 6, 7, 8, 9)


class MillhouseManastorm(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def minion_cost(self) -> int:
        return 2

    def refresh_cost(self) -> int:
        return 2

    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return (0, 6, 8, 9, 10, 11)


class CaptainEudora(Hero):
    power_cost = 1

    def hero_power_impl(self, context: BuyPhaseContext):
        self.digs_left -= 1
        if self.digs_left == 0:
            diggable_minions = [card for card in [minion() for minion in PrintingPress.all_types()] if card.tier <= context.owner.tavern_tier]
            random_minion = context.randomizer.select_gain_card(diggable_minions)
            random_minion.golden_transformation([])
            context.owner.gain_card(random_minion)
            self.digs_left = 5


class QueenWagtoggle(Hero):
    power_cost = 1

    def hero_power_impl(self, context: 'BuyPhaseContext'):
        for card in one_minion_per_type(context.owner.in_play, context.randomizer):
            card.attack += 2


class ForestWardenOmu(Hero):
    def hero_power_valid_impl(self, context: 'BuyPhaseContext'):
        return False

    def handle_event(self, event: 'CardEvent', context: 'BuyPhaseContext'):
        if event.event is EVENTS.TAVERN_UPGRADE:
            context.owner.coins = min(context.owner.coins + 2, 10)
