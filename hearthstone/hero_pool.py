from typing import Union

from hearthstone.cards import CardEvent
from hearthstone.events import BuyPhaseContext, CombatPhaseContext, COMBAT_START, SUMMON_COMBAT, BUY
from hearthstone.hero import Hero
from hearthstone.monster_types import DEMON, MECH


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