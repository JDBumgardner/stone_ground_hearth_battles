import sys
from inspect import getmembers, isclass
from typing import Union

from hearthstone.simulator.core.cards import MonsterCard
from hearthstone.simulator.core.events import CardEvent, EVENTS, BuyPhaseContext, CombatPhaseContext
from hearthstone.simulator.core.monster_types import MONSTER_TYPES


class FloatingWatcher(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.DEMON
    pool = MONSTER_TYPES.DEMON
    base_attack = 4
    base_health = 4
    mana_cost = 5

    def handle_event_powers(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.PLAYER_DAMAGED:
            bonus = 4 if self.golden else 2
            self.attack += bonus
            self.health += bonus


class ElistraTheImmortal(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.NEUTRAL
    base_attack = 4
    base_health = 4
    base_divine_shield = True
    base_reborn = True
    divert_taunt_attack = True
    legendary = True


class BarrensBlacksmith(MonsterCard):
    tier = 3
    monster_type = None
    base_attack = 3
    base_health = 5

    def frenzy(self, context: CombatPhaseContext):
        bonus = 4 if self.golden else 2
        for card in context.friendly_war_party.board:
            if card != self:
                card.attack += bonus
                card.health += bonus


class Siegebreaker(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.DEMON
    pool = MONSTER_TYPES.DEMON
    base_attack = 5
    base_health = 8
    base_taunt = True
    mana_cost = 7

    def handle_event_powers(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.COMBAT_PREPHASE or (event.event is EVENTS.SUMMON_COMBAT and event.card == self):
            demons = [card for card in context.friendly_war_party.board if
                      card != self and card.check_type(MONSTER_TYPES.DEMON)]
            for demon in demons:
                demon.attack += bonus
        elif event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board \
                and event.card != self and event.card.check_type(MONSTER_TYPES.DEMON):
            event.card.attack += bonus
        elif event.event is EVENTS.DIES and event.card == self:
            demons = [card for card in context.friendly_war_party.board if
                      card != self and card.check_type(MONSTER_TYPES.DEMON)]
            for demon in demons:
                demon.attack -= bonus


REMOVED_CARDS = [member[1] for member in getmembers(sys.modules[__name__],
                                                       lambda member: isclass(member) and issubclass(member,
                                                                                                     MonsterCard) and member.__module__ == __name__)]
