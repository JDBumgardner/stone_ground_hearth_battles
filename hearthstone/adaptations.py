from typing import Generator, Type

from hearthstone.cards import MonsterCard
from hearthstone.events import CombatPhaseContext


class Plant(MonsterCard):  # TODO: This was the only way I could find to get around circular imports. Any ideas for a better implementation?
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
        return True


class CracklingShield(Adaptation):
    def apply(self, card: 'MonsterCard'):
        card.divine_shield = True

    @classmethod
    def valid(cls, card: 'MonsterCard') -> bool:
        return not card.divine_shield


class FlamingClaws(Adaptation):
    def apply(self, card: 'MonsterCard'):
        card.attack += 3


class LivingSpores(Adaptation):
    def apply(self, card: 'MonsterCard'):
        def deathrattle(self, context: 'CombatPhaseContext'):
            summon_index = context.friendly_war_party.get_index(self)
            for i in range(2 * context.summon_minion_multiplier()):
                plant = Plant()
                if self.golden:
                    plant.golden_transformation([])
                context.friendly_war_party.summon_in_combat(plant, context, summon_index + i + 1)
        card.deathrattles.append(deathrattle)


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


class RockyCarapace(Adaptation):
    def apply(self, card: 'MonsterCard'):
        card.health += 3


class PoisonSpit(Adaptation):
    def apply(self, card: 'MonsterCard'):
        card.poisonous = True

    @classmethod
    def valid(cls, card: 'MonsterCard') -> bool:
        return not card.poisonous


def generate_valid_adaptations(card: 'MonsterCard') -> Generator['Type', None, None]:
    return (adaptation for adaptation in generate_all_adaptations() if adaptation.valid(card))


def generate_all_adaptations() -> Generator['Type', None, None]:
    yield CracklingShield
    yield FlamingClaws
    yield LivingSpores
    yield LightningSpeed
    yield Massive
    yield VolcanicMight
    yield RockyCarapace
    yield PoisonSpit