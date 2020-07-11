from typing import Union

from hearthstone.cards import CardEvent
from hearthstone.events import BuyPhaseContext, CombatPhaseContext, COMBAT_START, SUMMON_COMBAT, BUY
from hearthstone.hero import Hero
from hearthstone.monster_types import DEMON, MECH, PIRATE


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
            if minion.monster_type == DEMON:
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
        if event.event == COMBAT_START:
            if self.hero_power_used:
                for card in context.enemy_war_party.board:
                    card.take_damage(1)
                    card.resolve_death(context)


class Deathwing(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == COMBAT_START:
            all_minions = context.friendly_war_party.board + context.enemy_war_party.board
            for minion in all_minions:
                minion.attack += 2

        if event.event == SUMMON_COMBAT:
            event.card.attack += 2


class MillificentManastorm(Hero):
    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return False

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == BUY:
            if event.card.monster_type == MECH:
                event.card.attack += 1
                event.card.health += 1


class YoggSaron(Hero):
    power_cost = 2

    def hero_power_impl(self, context: BuyPhaseContext):
        card = context.randomizer.select_from_store(context.owner.store)
        card.attack += 1
        card.health += 1
        context.owner.store.remove(card)
        context.owner.hand.append(card)
        context.owner.check_golden(type(card))

    def hero_power_valid_impl(self, context: BuyPhaseContext):
        if not context.owner.room_in_hand():
            return False

        if not context.owner.store:
            return False

        return True


class PatchesThePirate(Hero):
    power_cost = 4

    def handle_event(self, event: CardEvent, context: BuyPhaseContext):
        if event.event == BUY and event.card.monster_type == PIRATE:
            self.power_cost = max(0, self.power_cost - 1)

    def hero_power_impl(self, context: BuyPhaseContext):
        pirates = [card for card in context.owner.tavern.deck.cards if
                   card.monster_type == PIRATE and card.tier <= context.owner.tavern_tier]

        card = context.randomizer.select_gain_card(pirates)
        context.owner.tavern.deck.cards.remove(card)
        context.owner.hand.append(card)
        context.owner.check_golden(type(card))
        self.power_cost = 4

    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return context.owner.room_in_hand()
