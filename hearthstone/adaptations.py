from inspect import isclass
from typing import Type, List

from hearthstone.cards import MonsterCard
from hearthstone.events import CombatPhaseContext


class Plant(MonsterCard):
    tier = 1
    monster_type = None
    base_attack = 1
    base_health = 1
    token = True


class Adaptation:
    def apply(self, card: 'MonsterCard'):
        pass

    @classmethod
    def valid(cls, card: 'MonsterCard') -> bool:
        pass


class AdaptBuff:

    class CracklingShield(Adaptation):
        def apply(self, card: 'MonsterCard'):
            card.divine_shield = True

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return not card.divine_shield

    class FlamingClaws(Adaptation):
        def apply(self, card: 'MonsterCard'):
            card.attack += 3

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return True

    class LivingSpores(Adaptation):
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

    class LightningSpeed(Adaptation):
        def apply(self, card: 'MonsterCard'):
            card.windfury = True

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return not card.windfury

    class Massive(Adaptation):
        def apply(self, card: 'MonsterCard'):
            card.taunt = True

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return not card.taunt

    class VolcanicMight(Adaptation):
        def apply(self, card: 'MonsterCard'):
            card.attack += 1
            card.health += 1

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return True

    class RockyCarapace(Adaptation):
        def apply(self, card: 'MonsterCard'):
            card.health += 3

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return True

    class PoisonSpit(Adaptation):
        def apply(self, card: 'MonsterCard'):
            card.poisonous = True

        @classmethod
        def valid(cls, card: 'MonsterCard') -> bool:
            return not card.poisonous


def valid_adaptations(card: 'MonsterCard') -> List['Type']:
    return [adaptation for adaptation in all_adaptations() if adaptation.valid(card)]


def all_adaptations() -> List['Type']:
    return [adaptation for adaptation in AdaptBuff.__dict__.values() if isclass(adaptation)]



