from typing import Union

from hearthstone.cards import CardEvent
from hearthstone.events import BuyPhaseContext, CombatPhaseContext, COMBAT_START
from hearthstone.hero import Hero
from hearthstone.monster_types import DEMON


class Pyramad(Hero):
    power_cost = 1

    def hero_power(self, context: BuyPhaseContext):
        if context.owner.coins < self.power_cost:
            return
        context.owner.coins -= self.power_cost
        if context.owner.in_play:
            minion = context.randomizer.select_friendly_minion(context.owner.in_play)
            minion.health += 4


class LordJaraxxus(Hero):
    power_cost = 1

    def hero_power(self, context: BuyPhaseContext):
        if context.owner.coins < self.power_cost:
            return
        for minion in context.owner.in_play:
            if minion.monster_type == DEMON:
                minion.attack += 1
                minion.health += 1


class PatchWerk(Hero):
    def starting_health(self) -> int:
        return 50


class Nefarian(Hero):
    power_cost = 1

    def __init__(self):
        self.nefarious_fire = False

    def hero_power(self, context: BuyPhaseContext):
        if context.owner.coins < self.power_cost:
            return
        context.owner.coins -= self.power_cost
        self.nefarious_fire = True

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == COMBAT_START:
            if self.nefarious_fire:
                for card in context.enemy_war_party.board:
                    card.take_damage(1)
                    card.resolve_death(context)
