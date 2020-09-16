from inspect import isclass
from typing import Generator, Type, List

from hearthstone.cards import MonsterCard
from hearthstone.events import CombatPhaseContext


class Plant(MonsterCard):  # TODO: This was the only way I could find to get around circular imports. Any ideas for a better implementation?
    tier = 1
    monster_type = None
    base_attack = 1
    base_health = 1
    token = True


class Adaptation:

    class CracklingShield:
        def apply(self, card: 'MonsterCard'):
            card.divine_shield = True

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return not card.divine_shield

    class FlamingClaws:
        def apply(self, card: 'MonsterCard'):
            card.attack += 3

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return True

    class LivingSpores:
        def apply(self, card: 'MonsterCard'):
            def deathrattle(self, context: 'CombatPhaseContext'):
                summon_index = context.friendly_war_party.get_index(self)
                for i in range(2 * context.summon_minion_multiplier()):
                    plant = Plant()
                    context.friendly_war_party.summon_in_combat(plant, context, summon_index + i + 1)
            card.deathrattles.append(deathrattle)

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return True

    class LightningSpeed:
        def apply(self, card: 'MonsterCard'):
            card.windfury = True

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return not card.windfury

    class Massive:
        def apply(self, card: 'MonsterCard'):
            card.taunt = True

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return not card.taunt

    class VolcanicMight:
        def apply(self, card: 'MonsterCard'):
            card.attack += 1
            card.health += 1

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return True

    class RockyCarapace:
        def apply(self, card: 'MonsterCard'):
            card.health += 3

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return True

    class PoisonSpit:
        def apply(self, card: 'MonsterCard'):
            card.poisonous = True

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return not card.poisonous


def valid_adaptations(card: 'MonsterCard') -> List['Type']:
    return [adaptation for adaptation in all_adaptations() if adaptation.valid(card)]


def all_adaptations() -> List['Type']:
    return [adaptation for adaptation in Adaptation.__dict__.values() if isclass(adaptation)]



