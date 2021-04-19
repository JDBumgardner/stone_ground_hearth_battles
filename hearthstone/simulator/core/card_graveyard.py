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
    monster_type = None
    base_attack = 4
    base_health = 4
    base_divine_shield = True
    base_reborn = True
    divert_taunt_attack = True
    legendary = True


REMOVED_CARDS = [member[1] for member in getmembers(sys.modules[__name__],
                                                       lambda member: isclass(member) and issubclass(member,
                                                                                                     MonsterCard) and member.__module__ == __name__)]
