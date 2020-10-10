import logging
import types
from typing import Union, List

from hearthstone.simulator.core import combat
from hearthstone.simulator.core.adaptations import valid_adaptations
from hearthstone.simulator.core.cards import MonsterCard, PrintingPress, one_minion_per_type
from hearthstone.simulator.core.events import BuyPhaseContext, CombatPhaseContext, EVENTS, CardEvent
from hearthstone.simulator.core.monster_types import MONSTER_TYPES


class MamaBear(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 4
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if (event.event is EVENTS.SUMMON_BUY or (
                event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board)) and event.card.check_type(
                MONSTER_TYPES.BEAST) and event.card != self:
            bonus = 8 if self.golden else 4
            event.card.attack += bonus
            event.card.health += bonus


class ShifterZerus(MonsterCard):
    tier = 3
    monster_type = None
    base_attack = 1
    base_health = 1
    legendary = True

    def handle_event_in_hand(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.BUY_START:
            # TODO: can this turn into a token?
            self.zerus_shift(context)


class SneedsOldShredder(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.MECH
    base_attack = 5
    base_health = 7

    def base_deathrattle(self, context: CombatPhaseContext):
        count = 2 if self.golden else 1
        summon_index = context.friendly_war_party.get_index(self)
        i = 0
        for _ in range(count):
            for _ in range(context.summon_minion_multiplier()):
                legendary_minions = [card_type for card_type in PrintingPress.all_types() if card_type.legendary]
                random_minion = context.randomizer.select_summon_minion(legendary_minions)()
                context.friendly_war_party.summon_in_combat(random_minion, context, summon_index + i + 1)
                i += 1


class FreedealingGambler(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 3
    base_health = 3
    redeem_rate = 3

    def golden_transformation(self, base_cards: List['MonsterCard']):
        super().golden_transformation(base_cards)
        self.redeem_rate *= 2


class DragonspawnLieutenant(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 2
    base_health = 3
    base_taunt = True
    mana_cost = 2


class RighteousProtector(MonsterCard):
    tier = 1
    monster_type = None
    base_attack = 1
    base_health = 1
    base_taunt = True
    base_divine_shield = True


class AlleyCat(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 1
    base_health = 1

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        for _ in range(context.summon_minion_multiplier()):
            tabby_cat = TabbyCat()
            if self.golden:
                tabby_cat.golden_transformation([])
            context.owner.summon_from_void(tabby_cat)


class TabbyCat(MonsterCard):
    token = True
    tier = 1
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 1
    base_health = 1


class ScavengingHyena(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 2
    base_health = 2
    mana_cost = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.DIES and event.card.check_type(
                MONSTER_TYPES.BEAST) and event.card in context.friendly_war_party.board:
            self.attack += 4 if self.golden else 2
            self.health += 2 if self.golden else 1


class FiendishServant(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 2
    base_health = 1

    def base_deathrattle(self, context: CombatPhaseContext):
        count = 2 if self.golden else 1
        for _ in range(count):
            friendly_monsters = [card for card in context.friendly_war_party.board if card != self and not card.dead]
            if friendly_monsters:
                friendly_monster = context.randomizer.select_friendly_minion(friendly_monsters)
                friendly_monster.attack += self.attack


class WrathWeaver(MonsterCard):
    tier = 1
    monster_type = None
    base_attack = 1
    base_health = 1

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.SUMMON_BUY and event.card.check_type(MONSTER_TYPES.DEMON):
            bonus = 4 if self.golden else 2
            context.owner.take_damage(1)
            self.attack += bonus
            self.health += bonus


class MicroMachine(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MECH
    base_attack = 1
    base_health = 2
    mana_cost = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.BUY_START:
            if self.golden:
                self.attack += 2
            else:
                self.attack += 1


class MurlocTidecaller(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 1
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        friendly_summon = event.event is EVENTS.SUMMON_BUY or (
                event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board)
        if friendly_summon and event.card.check_type(MONSTER_TYPES.MURLOC) and event.card != self:
            self.attack += bonus


class MurlocTidehunter(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 2
    base_health = 1
    mana_cost = 2

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        for _ in range(context.summon_minion_multiplier()):
            murloc_scout = MurlocScout()
            if self.golden:
                murloc_scout.golden_transformation([])
            context.owner.summon_from_void(murloc_scout)


class MurlocScout(MonsterCard):
    token = True
    tier = 1
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 1
    base_health = 1


class SelflessHero(MonsterCard):
    tier = 1
    monster_type = None
    base_attack = 2
    base_health = 1

    def base_deathrattle(self, context: CombatPhaseContext):
        friendly_minions = [card for card in context.friendly_war_party.board if
                            card != self and not card.dead and not card.divine_shield]

        num_friendly_shields = 2 if self.golden else 1
        for _ in range(num_friendly_shields):
            if friendly_minions:
                random_friendly_minion = context.randomizer.select_friendly_minion(friendly_minions)
                random_friendly_minion.divine_shield = True
                friendly_minions.remove(random_friendly_minion)


class VulgarHomunculus(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 2
    base_health = 4
    base_taunt = True
    mana_cost = 2

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        context.owner.take_damage(2)


class RedWhelp(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 1
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.COMBAT_START:
            num_friendly_dragons = len([card for card in context.friendly_war_party.board if
                                        not card.dead and card.check_type(MONSTER_TYPES.DRAGON)])
            targets = [card for card in context.enemy_war_party.board if not card.is_dying()]
            if not targets:
                return
            num_damage_instances = 2 if self.golden else 1
            for _ in range(num_damage_instances):
                target = context.randomizer.select_enemy_minion(targets)
                target.take_damage(num_friendly_dragons, context.enemy_context(), self)
                target.resolve_death(context.enemy_context(), self)  # TODO: Order of death resolution?


class HarvestGolem(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.MECH
    base_attack = 2
    base_health = 3

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(context.summon_minion_multiplier()):
            damaged_golem = DamagedGolem()
            if self.golden:
                damaged_golem.golden_transformation([])
            context.friendly_war_party.summon_in_combat(damaged_golem, context, summon_index + i + 1)


class DamagedGolem(MonsterCard):
    token = True
    tier = 1
    monster_type = MONSTER_TYPES.MECH
    base_attack = 2
    base_health = 1


class KaboomBot(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.MECH
    base_attack = 2
    base_health = 2

    def base_deathrattle(self, context: CombatPhaseContext):
        num_damage_instances = 2 if self.golden else 1
        for _ in range(num_damage_instances):
            targets = [card for card in context.enemy_war_party.board if not card.is_dying()]
            if not targets:
                break
            target = context.randomizer.select_enemy_minion(targets)
            target.take_damage(4, context.enemy_context(), self)
            target.resolve_death(context.enemy_context(), self)  # TODO: Order of death resolution?


class KindlyGrandmother(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 1
    base_health = 1
    mana_cost = 2

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(context.summon_minion_multiplier()):
            big_bad_wolf = BigBadWolf()
            if self.golden:
                big_bad_wolf.golden_transformation([])
            context.friendly_war_party.summon_in_combat(big_bad_wolf, context, summon_index + i + 1)


class BigBadWolf(MonsterCard):
    token = True
    tier = 1
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 3
    base_health = 2


class MetaltoothLeaper(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.MECH
    base_attack = 3
    base_health = 3

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        for card in context.owner.in_play:
            if card != self and card.check_type(MONSTER_TYPES.MECH):
                bonus = 4 if self.golden else 2
                card.attack += bonus


class RabidSaurolisk(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 4
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.SUMMON_BUY and event.card.deathrattles:
            self.attack += bonus
            self.health += bonus


class GlyphGuardian(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 2
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.ON_ATTACK and event.card == self:
            multiplier = 2
            if self.golden:
                multiplier = 3
            self.attack *= multiplier


class Imprisoner(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 3
    base_health = 3
    base_taunt = True

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(context.summon_minion_multiplier()):
            imp = Imp()
            if self.golden:
                imp.golden_transformation([])
            context.friendly_war_party.summon_in_combat(imp, context, summon_index + i + 1)


class Imp(MonsterCard):
    token = True
    tier = 1
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 1
    base_health = 1


class MurlocWarleader(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 3
    base_health = 3

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 4 if self.golden else 2
        if event.event is EVENTS.COMBAT_START or (event.event is EVENTS.SUMMON_COMBAT and event.card == self):
            murlocs = [card for card in context.friendly_war_party.board if
                       card != self and card.check_type(MONSTER_TYPES.MURLOC)]
            for murloc in murlocs:
                murloc.attack += bonus
        elif event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board \
                and event.card != self and event.card.check_type(MONSTER_TYPES.MURLOC):
            event.card.attack += bonus
        elif event.event is EVENTS.DIES and event.card == self:
            murlocs = [card for card in context.friendly_war_party.board if
                       card != self and card.check_type(MONSTER_TYPES.MURLOC)]
            for murloc in murlocs:
                murloc.attack -= bonus


class StewardOfTime(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 3
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.SELL and event.card == self:
            for card in context.owner.store:
                card.attack += bonus
                card.health += bonus


class Scallywag(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 2
    base_health = 1

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(context.summon_minion_multiplier()):
            pirate_summon = SkyPirate()
            if self.golden:
                pirate_summon.golden_transformation([])
            context.friendly_war_party.summon_in_combat(pirate_summon, context, summon_index + i + 1)


class SkyPirate(MonsterCard):
    tier = 1
    token = True
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 1
    base_health = 1

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SUMMON_COMBAT and event.card == self:
            attacking_war_party = context.friendly_war_party
            defending_war_party = context.enemy_war_party
            defender = defending_war_party.get_attack_target(context.randomizer, self)
            if not defender:
                return
            logging.debug(f'{attacking_war_party.owner.name} is attacking {defending_war_party.owner.name}')
            combat.start_attack(self, defender, attacking_war_party, defending_war_party, context.randomizer)


class DeckSwabbie(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 2
    base_health = 2

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        discount = 2 if self.golden else 1
        context.owner.tavern_upgrade_cost -= discount


class UnstableGhoul(MonsterCard):
    tier = 2
    monster_type = None
    base_attack = 1
    base_health = 3
    base_taunt = True
    mana_cost = 2

    def base_deathrattle(self, context: CombatPhaseContext):
        all_minions = [card for card in context.friendly_war_party.board + context.enemy_war_party.board if
                       not card.dead]
        count = 2 if self.golden else 1
        for _ in range(count):
            for minion in all_minions:
                if minion.is_dying():
                    continue
                if minion in context.friendly_war_party.board:
                    minion.take_damage(1, context, self)
                    minion.resolve_death(context, self)  # TODO: Order of death resolution?
                elif minion in context.enemy_war_party.board:
                    minion.take_damage(1, context.enemy_context(), self)
                    minion.resolve_death(context.enemy_context(), self)


class RockpoolHunter(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 2
    base_health = 3
    num_battlecry_targets = [1]
    mana_cost = 2

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def valid_battlecry_target(self, card: MonsterCard) -> bool:
        return card.check_type(MONSTER_TYPES.MURLOC) and card != self


class RatPack(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 2
    base_health = 2

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(self.attack):
            rat = Rat()
            if self.golden:
                rat.golden_transformation([])
            context.friendly_war_party.summon_in_combat(rat, context, summon_index + i + 1)


class Rat(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 1
    base_health = 1
    token = True


class NathrezimOverseer(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 2
    base_health = 3
    num_battlecry_targets = [1]

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def valid_battlecry_target(self, card: MonsterCard) -> bool:
        return card.check_type(MONSTER_TYPES.DEMON) and card != self


class OldMurkeye(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 2
    base_health = 4
    legendary = True

    # charge has no effect in battlegrounds

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.COMBAT_START:
            self.attack += bonus * sum(
                1 for murloc in context.friendly_war_party.board + context.enemy_war_party.board if
                murloc.check_type(MONSTER_TYPES.MURLOC) and event.card != self)
        if event.event is EVENTS.DIES and event.card in context.friendly_war_party.board + context.enemy_war_party.board and event.card.check_type(
                MONSTER_TYPES.MURLOC):
            self.attack -= bonus
        if event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board + context.enemy_war_party.board and event.card.check_type(
                MONSTER_TYPES.MURLOC):
            self.attack += bonus


class CrystalWeaver(MonsterCard):
    tier = 3
    base_attack = 5
    base_health = 4

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        for card in context.owner.in_play:
            if card.check_type(MONSTER_TYPES.DEMON):
                card.attack += bonus
                card.health += bonus


class MechanoEgg(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.MECH
    base_attack = 0
    base_health = 5

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(context.summon_minion_multiplier()):
            robosaur = Robosaur()
            if self.golden:
                robosaur.golden_transformation([])
            context.friendly_war_party.summon_in_combat(robosaur, context, summon_index + i + 1)


class Robosaur(MonsterCard):
    token = True
    tier = 1
    monster_type = MONSTER_TYPES.MECH
    base_attack = 8
    base_health = 8


class PogoHopper(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.MECH
    base_attack = 1
    base_health = 1
    tracked = True

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        self.attack += context.owner.counted_cards[type(self)] * bonus
        self.health += context.owner.counted_cards[type(self)] * bonus


class Goldgrubber(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 2
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.BUY_END:
            bonus = 4 if self.golden else 2
            for card in context.owner.in_play:
                if card.golden:
                    self.attack += bonus
                    self.health += bonus


class SpawnOfNzoth(MonsterCard):
    tier = 2
    base_attack = 2
    base_health = 2

    def base_deathrattle(self, context: CombatPhaseContext):
        bonus = 2 if self.golden else 1
        for card in context.friendly_war_party.board:
            card.attack += bonus
            card.health += bonus


class BloodsailCannoneer(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 4
    base_health = 2

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 6 if self.golden else 3
        for card in context.owner.in_play:
            if card.check_type(MONSTER_TYPES.PIRATE) and card != self:
                card.attack += bonus


class ColdlightSeer(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 2
    base_health = 3

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        for card in context.owner.in_play:
            if card.check_type(MONSTER_TYPES.MURLOC) and card != self:
                card.health += bonus


class CrowdFavorite(MonsterCard):
    tier = 3
    base_attack = 4
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.SUMMON_BUY and event.card.battlecry:
            self.attack += bonus
            self.health += bonus


class DeflectOBot(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.MECH
    base_attack = 3
    base_health = 2
    base_divine_shield = True

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.SUMMON_COMBAT and event.card.check_type(
                MONSTER_TYPES.MECH) and event.card in context.friendly_war_party.board:
            self.attack += bonus
            self.divine_shield = True


class FelfinNavigator(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 4
    base_health = 4

    def base_battlecry(self, targets: List[MonsterCard], context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        bonus = 2 if self.golden else 1
        for card in context.owner.in_play:
            if card.check_type(MONSTER_TYPES.MURLOC) and card != self:
                card.health += bonus
                card.attack += bonus


class Houndmaster(MonsterCard):
    tier = 3
    base_attack = 4
    base_health = 3
    num_battlecry_targets = [1]

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus
            targets[0].taunt = True

    def valid_battlecry_target(self, card: MonsterCard) -> bool:
        return card.check_type(MONSTER_TYPES.BEAST)


class ImpGangBoss(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 2
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.CARD_DAMAGED and self == event.card:
            summon_index = context.friendly_war_party.get_index(self)
            for i in range(context.summon_minion_multiplier()):
                imp = Imp()
                if self.golden:
                    imp.golden_transformation([])
                context.friendly_war_party.summon_in_combat(imp, context, summon_index + i + 1)


class InfestedWolf(MonsterCard):
    tier = 3
    base_attack = 3
    base_health = 3
    monster_type = MONSTER_TYPES.BEAST

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(2 * context.summon_minion_multiplier()):
            spider = Spider()
            if self.golden:
                spider.golden_transformation([])
            context.friendly_war_party.summon_in_combat(spider, context, summon_index + i + 1)


class Spider(MonsterCard):
    tier = 1
    base_attack = 1
    base_health = 1
    token = True
    monster_type = MONSTER_TYPES.BEAST


class MonstrousMacaw(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 4
    base_health = 3

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.AFTER_ATTACK_DAMAGE and self == event.card:
            deathrattle_triggers = 2 if self.golden else 1
            for _ in range(deathrattle_triggers):
                friendly_deathrattlers = [card for card in context.friendly_war_party.board if
                                          card != self and not card.dead and card.deathrattles]
                if friendly_deathrattlers:
                    deathrattler = context.randomizer.select_friendly_minion(friendly_deathrattlers)
                    for deathrattle in deathrattler.deathrattles:
                        for _ in range(context.deathrattle_multiplier()):
                            deathrattle(deathrattler, context)


class ScrewjankClunker(MonsterCard):
    tier = 3
    base_attack = 2
    base_health = 5
    monster_type = MONSTER_TYPES.MECH
    num_battlecry_targets = [1]

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def valid_battlecry_target(self, card: MonsterCard) -> bool:
        return card.check_type(MONSTER_TYPES.MECH)


class PackLeader(MonsterCard):
    tier = 2
    base_attack = 2
    base_health = 3
    monster_type = None

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        friendly_summon = event.event is EVENTS.SUMMON_BUY or (
                event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board)
        if friendly_summon and event.card.check_type(MONSTER_TYPES.BEAST) and event.card != self:
            bonus = 4 if self.golden else 2
            event.card.attack += bonus


class PilotedShredder(MonsterCard):
    tier = 3
    base_attack = 4
    base_health = 3
    monster_type = MONSTER_TYPES.MECH

    def base_deathrattle(self, context: CombatPhaseContext):
        count = 2 if self.golden else 1
        summon_index = context.friendly_war_party.get_index(self)
        i = 0
        for _ in range(count):
            for _ in range(context.summon_minion_multiplier()):
                two_cost_minions = [card_type for card_type in PrintingPress.all_types() if card_type.mana_cost == 2]
                random_minion = context.randomizer.select_summon_minion(two_cost_minions)()
                context.friendly_war_party.summon_in_combat(random_minion, context, summon_index + i + 1)
                i += 1


class SaltyLooter(MonsterCard):
    tier = 3
    base_attack = 3
    base_health = 3
    monster_type = MONSTER_TYPES.PIRATE

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SUMMON_BUY and event.card.check_type(MONSTER_TYPES.PIRATE) and event.card != self:
            bonus = 2 if self.golden else 1
            self.attack += bonus
            self.health += bonus


class SoulJuggler(MonsterCard):
    tier = 3
    base_attack = 3
    base_health = 3
    monster_type = None

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIES and event.card.check_type(
                MONSTER_TYPES.DEMON) and event.card in context.friendly_war_party.board:
            count = 2 if self.golden else 1
            for _ in range(count):
                targets = [card for card in context.enemy_war_party.board if not card.is_dying()]
                if targets:
                    target = context.randomizer.select_enemy_minion(targets)
                    target.take_damage(3, context.enemy_context(), self)
                    target.resolve_death(context.enemy_context(), self)  # TODO: Order of death resolution?


class TwilightEmissary(MonsterCard):
    tier = 3
    base_attack = 4
    base_health = 4
    monster_type = MONSTER_TYPES.DRAGON
    base_taunt = True
    num_battlecry_targets = [1]

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def valid_battlecry_target(self, card: MonsterCard) -> bool:
        return card.check_type(MONSTER_TYPES.DRAGON)


class Khadgar(MonsterCard):
    tier = 3
    base_attack = 2
    base_health = 2
    monster_type = None
    legendary = True
    mana_cost = 2

    def summon_minion_multiplier(self) -> int:
        return 3 if self.golden else 2


class SavannahHighmane(MonsterCard):
    tier = 4
    base_attack = 6
    base_health = 5
    monster_type = MONSTER_TYPES.BEAST

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(2 * context.summon_minion_multiplier()):
            hyena = Hyena()
            if self.golden:
                hyena.golden_transformation([])
            context.friendly_war_party.summon_in_combat(hyena, context, summon_index + i + 1)


class Hyena(MonsterCard):
    tier = 1
    base_attack = 2
    base_health = 2
    monster_type = MONSTER_TYPES.BEAST
    token = True


class SecurityRover(MonsterCard):
    tier = 4
    base_attack = 2
    base_health = 6
    monster_type = MONSTER_TYPES.MECH

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.CARD_DAMAGED and self == event.card:
            summon_index = context.friendly_war_party.get_index(self)
            for i in range(context.summon_minion_multiplier()):
                bot = GuardBot()
                if self.golden:
                    bot.golden_transformation([])
                context.friendly_war_party.summon_in_combat(bot, context, summon_index + i + 1)


class GuardBot(MonsterCard):
    tier = 1
    base_attack = 2
    base_health = 3
    monster_type = MONSTER_TYPES.MECH
    token = True
    base_taunt = True


class VirmenSensei(MonsterCard):
    tier = 4
    base_attack = 4
    base_health = 5
    monster_type = None
    num_battlecry_targets = [1]

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def valid_battlecry_target(self, card: MonsterCard) -> bool:
        return card.check_type(MONSTER_TYPES.BEAST)


class RipsnarlCaptain(MonsterCard):
    tier = 4
    base_attack = 3
    base_health = 4
    monster_type = MONSTER_TYPES.PIRATE

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.ON_ATTACK and event.card.check_type(
                MONSTER_TYPES.PIRATE) and event.card in context.friendly_war_party.board and event.card != self:
            bonus = 4 if self.golden else 2
            event.card.attack += bonus
            event.card.health += bonus


class DefenderOfArgus(MonsterCard):
    tier = 4
    base_attack = 2
    base_health = 3
    monster_type = None
    num_battlecry_targets = [1, 2]

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        if targets:
            bonus = 2 if self.golden else 1
            for i in range(len(targets)):
                targets[i].attack += bonus
                targets[i].health += bonus
                targets[i].taunt = True


class SouthseaCaptain(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 3
    base_health = 3

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.COMBAT_START or (event.event is EVENTS.SUMMON_COMBAT and event.card == self):
            pirates = [card for card in context.friendly_war_party.board if
                       card != self and card.check_type(MONSTER_TYPES.PIRATE)]
            for pirate in pirates:
                pirate.attack += bonus
                pirate.health += bonus
        elif event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board \
                and event.card != self and event.card.check_type(MONSTER_TYPES.PIRATE):
            event.card.attack += bonus
            event.card.health += bonus
        elif event.event is EVENTS.DIES and event.card == self:
            pirates = [card for card in context.friendly_war_party.board if
                       card != self and card.check_type(MONSTER_TYPES.PIRATE)]
            for pirate in pirates:
                pirate.attack -= bonus
                if pirate.health > pirate.base_health > pirate.health - bonus:
                    pirate.health = pirate.base_health


class BolvarFireblood(MonsterCard):
    tier = 4
    monster_type = None
    base_attack = 1
    base_health = 7
    base_divine_shield = True
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIVINE_SHIELD_LOST and event.card in context.friendly_war_party.board:
            bonus = 4 if self.golden else 2
            self.attack += bonus


class DrakonidEnforcer(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 3
    base_health = 6

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIVINE_SHIELD_LOST and event.card in context.friendly_war_party.board:
            bonus = 4 if self.golden else 2
            self.attack += bonus
            self.health += bonus


class BronzeWarden(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 2
    base_health = 1
    base_divine_shield = True
    base_reborn = True


class Amalgam(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.ALL
    base_attack = 1
    base_health = 1
    token = True

    # TODO: this can't be sold on the first round


class ReplicatingMenace(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.MECH
    base_attack = 3
    base_health = 1
    base_magnetic = True

    def __init__(self):
        def base_deathrattle(card, context: 'CombatPhaseContext'):
            summon_index = context.friendly_war_party.get_index(card)
            for i in range(3 * context.summon_minion_multiplier()):
                microbot = Microbot()
                if self.golden:
                    microbot.golden_transformation([])
                context.friendly_war_party.summon_in_combat(microbot, context, summon_index + i + 1)
        self.base_deathrattle = types.MethodType(base_deathrattle, self)
        super().__init__()


class Microbot(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MECH
    base_attack = 1
    base_health = 1
    token = True


class Junkbot(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.MECH
    base_attack = 1
    base_health = 5

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIES and event.card in context.friendly_war_party.board and event.card.check_type(
                MONSTER_TYPES.MECH):
            bonus = 4 if self.golden else 2
            self.attack += bonus
            self.health += bonus


class StrongshellScavenger(MonsterCard):
    tier = 5
    monster_type = None
    base_attack = 2
    base_health = 3

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        taunt_minions = [card for card in context.owner.in_play if card.taunt]
        bonus = 4 if self.golden else 2
        for minion in taunt_minions:
            minion.attack += bonus
            minion.health += bonus


class Voidlord(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 3
    base_health = 9
    base_taunt = True

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(3 * context.summon_minion_multiplier()):
            demon = Demon()
            if self.golden:
                demon.golden_transformation([])
            context.friendly_war_party.summon_in_combat(demon, context, summon_index + i + 1)


class Demon(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 1
    base_health = 3
    base_taunt = True
    token = True


class AnnihilanBattlemaster(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 3
    base_health = 1

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        multiplier = 2 if self.golden else 1
        damage_taken = context.owner.hero.starting_health() - context.owner.health
        self.health += multiplier * damage_taken


class CapnHoggarr(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 6
    base_health = 6
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY and event.card.check_type(MONSTER_TYPES.PIRATE):
            gold = 2 if self.golden else 1
            context.owner.coins += gold


class KingBagurgle(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 6
    base_health = 3
    legendary = True

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        for card in context.owner.in_play:
            if card.check_type(MONSTER_TYPES.MURLOC) and card != self:
                bonus = 4 if self.golden else 2
                card.attack += bonus
                card.health += bonus

    def base_deathrattle(self, context: CombatPhaseContext):
        for card in context.friendly_war_party.board:
            bonus = 4 if self.golden else 2
            if card.check_type(MONSTER_TYPES.MURLOC) and card != self:
                card.attack += bonus
                card.health += bonus


class RazorgoreTheUntamed(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 2
    base_health = 4
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END:
            bonus = 2 if self.golden else 1
            for card in context.owner.in_play:
                if card.check_type(MONSTER_TYPES.DRAGON):
                    self.attack += bonus
                    self.health += bonus


class Ghastcoiler(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 7
    base_health = 7

    def base_deathrattle(self, context: CombatPhaseContext):
        count = 4 if self.golden else 2
        summon_index = context.friendly_war_party.get_index(self)
        i = 0
        for _ in range(count):
            for _ in range(context.summon_minion_multiplier()):
                deathrattlers = [card_type for card_type in PrintingPress.all_types() if
                                 card_type.base_deathrattle and card_type != type(self)]
                random_minion_type = context.randomizer.select_summon_minion(deathrattlers)
                context.friendly_war_party.summon_in_combat(random_minion_type(), context, summon_index + i + 1)
                i += 1


class DreadAdmiralEliza(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 6
    base_health = 7
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.ON_ATTACK and event.card in context.friendly_war_party.board and event.card.check_type(
                MONSTER_TYPES.PIRATE):
            bonus = 2 if self.golden else 1
            for card in context.friendly_war_party.board:
                card.attack += bonus
                card.health += bonus


class GoldrinnTheGreatWolf(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 4
    base_health = 4
    legendary = True

    def base_deathrattle(self, context: CombatPhaseContext):
        for card in context.friendly_war_party.board:
            if card.check_type(MONSTER_TYPES.BEAST):
                bonus = 10 if self.golden else 5
                card.attack += bonus
                card.health += bonus


class ImpMama(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 6
    base_health = 10

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.CARD_DAMAGED and event.card == self:
            count = 2 if self.golden else 1
            summon_index = context.friendly_war_party.get_index(self)
            i = 0
            for _ in range(count):
                for _ in range(context.summon_minion_multiplier()):
                    # TODO: can this summon tokens? Can this summon itself?
                    demons = [card_type for card_type in PrintingPress.all_types() if
                              card_type.check_type(MONSTER_TYPES.DEMON) and card_type != type(self)]
                    random_minion_type = context.randomizer.select_summon_minion(demons)
                    random_minion = random_minion_type()
                    random_minion.taunt = True
                    context.friendly_war_party.summon_in_combat(random_minion, context, summon_index + i + 1)
                    i += 1


class KalecgosArcaneAspect(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 4
    base_health = 12
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.SUMMON_BUY and event.card.battlecry:
            for card in context.owner.in_play:
                if card.check_type(MONSTER_TYPES.DRAGON):
                    bonus = 2 if self.golden else 1
                    card.attack += bonus
                    card.health += bonus


class NadinaTheRed(MonsterCard):
    tier = 6
    monster_type = None
    base_attack = 7
    base_health = 4
    legendary = True

    def base_deathrattle(self, context: CombatPhaseContext):
        for card in context.friendly_war_party.board:
            if card.check_type(MONSTER_TYPES.DRAGON):
                card.divine_shield = True


class TheTideRazor(MonsterCard):
    tier = 6
    monster_type = None
    base_attack = 6
    base_health = 4

    def base_deathrattle(self, context: CombatPhaseContext):
        count = 6 if self.golden else 3
        summon_index = context.friendly_war_party.get_index(self)
        i = 0
        for _ in range(count):
            for _ in range(context.summon_minion_multiplier()):
                # TODO: can this summon tokens?
                pirates = [card_type for card_type in PrintingPress.all_types() if card_type.check_type(MONSTER_TYPES.PIRATE)]
                random_minion_type = context.randomizer.select_summon_minion(pirates)
                context.friendly_war_party.summon_in_combat(random_minion_type(), context, summon_index + i + 1)
                i += 1


class Toxfin(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 1
    base_health = 2
    num_battlecry_targets = [1]

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        if targets:
            targets[0].poisonous = True

    def valid_battlecry_target(self, card: MonsterCard) -> bool:
        return card.check_type(MONSTER_TYPES.MURLOC)


class Maexxna(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 2
    base_health = 8
    base_poisonous = True
    legendary = True


class HeraldOfFlame(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 5
    base_health = 6

    def overkill(self, context: CombatPhaseContext):
        damage = 6 if self.golden else 3
        leftmost_index = 0
        while True:
            if not context.enemy_war_party.board[leftmost_index].is_dying():
                break
            leftmost_index += 1
            if leftmost_index >= len(context.enemy_war_party.board):
                return
        context.enemy_war_party.board[leftmost_index].take_damage(damage, context.enemy_context(), self)
        context.enemy_war_party.board[leftmost_index].resolve_death(context.enemy_context(), self)


class IronhideDirehorn(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 7
    base_health = 7

    def overkill(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(context.summon_minion_multiplier()):
            runt = IronhideRunt()
            if self.golden:
                runt.golden_transformation([])
            context.friendly_war_party.summon_in_combat(runt, context, summon_index + i + 1)


class IronhideRunt(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 5
    base_health = 5
    token = True


class NatPagleExtremeAngler(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 8
    base_health = 5
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.AFTER_ATTACK_DAMAGE and self == event.card and event.foe.is_dying():
            for _ in range(2 if self.golden else 1):
                if context.friendly_war_party.owner.room_in_hand():
                    random_minion = context.randomizer.select_gain_card(context.friendly_war_party.owner.tavern.deck.unique_cards())
                    context.friendly_war_party.owner.tavern.deck.remove_card(random_minion)
                    context.friendly_war_party.owner.gain_hand_card(random_minion)


class FloatingWatcher(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 4
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.PLAYER_DAMAGED:
            bonus = 4 if self.golden else 2
            self.attack += bonus
            self.health += bonus


class MalGanis(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 9
    base_health = 7
    give_immunity = True
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 4 if self.golden else 2
        if event.event is EVENTS.COMBAT_START or (event.event is EVENTS.SUMMON_COMBAT and event.card == self):
            demons = [card for card in context.friendly_war_party.board if
                      card != self and card.check_type(MONSTER_TYPES.DEMON)]
            for demon in demons:
                demon.attack += bonus
                demon.health += bonus
        elif event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board \
                and event.card != self and event.card.check_type(MONSTER_TYPES.DEMON):
            event.card.attack += bonus
            event.card.health += bonus
        elif event.event is EVENTS.DIES and event.card == self:
            demons = [card for card in context.friendly_war_party.board if
                      card != self and card.check_type(MONSTER_TYPES.DEMON)]
            for demon in demons:
                demon.attack -= bonus
                if demon.health > demon.base_health > demon.health - bonus:
                    demon.health = demon.base_health


class BaronRivendare(MonsterCard):
    tier = 5
    monster_type = None
    base_attack = 1
    base_health = 7
    legendary = True

    def deathrattle_multiplier(self) -> int:
        return 3 if self.golden else 2


class BrannBronzebeard(MonsterCard):
    tier = 5
    monster_type = None
    base_attack = 2
    base_health = 4
    legendary = True

    def battlecry_multiplier(self) -> int:
        return 3 if self.golden else 2


class IronSensei(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.MECH
    base_attack = 2
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END:
            friendly_mechs = [card for card in context.owner.in_play if
                              card.check_type(MONSTER_TYPES.MECH) and card != self]
            if friendly_mechs:
                mech = context.randomizer.select_friendly_minion(friendly_mechs)
                bonus = 4 if self.golden else 2
                mech.attack += bonus
                mech.health += bonus


class YoHoOgre(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 2
    base_health = 8
    base_taunt = True

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.AFTER_ATTACK_DEATHRATTLES and event.card == self and not self.is_dying():
            attacking_war_party = context.friendly_war_party
            defending_war_party = context.enemy_war_party
            defender = defending_war_party.get_attack_target(context.randomizer, self)
            if not defender:
                return
            logging.debug(f'{self} triggers after surviving an attack')
            combat.start_attack(self, defender, attacking_war_party, defending_war_party, context.randomizer)


class WaxriderTogwaggle(MonsterCard):
    tier = 2
    monster_type = None
    base_attack = 1
    base_health = 2
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.DIES and event.card in context.enemy_war_party.board and event.foe in context.friendly_war_party.board and event.foe.check_type(
                MONSTER_TYPES.DRAGON):
            bonus = 4 if self.golden else 2
            self.attack += bonus
            self.health += bonus


class HangryDragon(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 4
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.END_COMBAT and event.won_combat:
            bonus = 4 if self.golden else 2
            self.attack += bonus
            self.health += bonus


class LightfangEnforcer(MonsterCard):
    tier = 5
    monster_type = None
    base_attack = 2
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END:
            for card in one_minion_per_type(context.owner.in_play, context.randomizer):
                card.attack += 4 if self.golden else 2
                card.health += 2 if self.golden else 1


class MenagerieMug(MonsterCard):
    tier = 2
    monster_type = None
    base_attack = 2
    base_health = 2

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        available_minions = one_minion_per_type(context.owner.in_play, context.randomizer)
        for _ in range(3):
            if available_minions:
                card = context.randomizer.select_friendly_minion(available_minions)
                available_minions.remove(card)
                card.attack += bonus
                card.health += bonus


class MenagerieJug(MonsterCard):
    tier = 4
    monster_type = None
    base_attack = 3
    base_health = 3

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        available_minions = one_minion_per_type(context.owner.in_play, context.randomizer)
        for _ in range(3):
            if available_minions:
                card = context.randomizer.select_friendly_minion(available_minions)
                available_minions.remove(card)
                card.attack += bonus
                card.health += bonus


class MicroMummy(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MECH
    base_attack = 1
    base_health = 2
    base_reborn = True
    mana_cost = 2

    def handle_event_powers(self, event: CardEvent, context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END:
            other_minions = [card for card in context.owner.in_play if card != self]
            if other_minions:
                card = context.randomizer.select_friendly_minion(other_minions)
                bonus = 2 if self.golden else 1
                card.attack += bonus


class KangorsApprentice(MonsterCard):
    tier = 6
    monster_type = None
    base_attack = 3
    base_health = 6

    def base_deathrattle(self, context: CombatPhaseContext): #TODO does this get tokens?
        count = 4 if self.golden else 2
        summon_index = context.friendly_war_party.get_index(self)
        dead_mechs = [dead_minion for dead_minion in context.friendly_war_party.dead_minions if
                          dead_minion.check_type(MONSTER_TYPES.MECH)]
        for index, mech in enumerate(dead_mechs[:count]):
            summon_minion = mech.unbuffed_copy()
            context.friendly_war_party.summon_in_combat(summon_minion, context, summon_index + index + 1)


class ZappSlywick(MonsterCard):
    tier = 6
    monster_type = None
    base_attack = 7
    base_health = 10
    base_windfury = True
    legendary = True

    def golden_transformation(self, base_cards: List['MonsterCard']):
        super().golden_transformation(base_cards)
        self.windfury = False
        self.mega_windfury = True

    def valid_attack_targets(self, live_enemies: List['MonsterCard']) -> List['MonsterCard']:
        if self.attack <= 0 or not live_enemies:
            return []
        else:
            min_attack = min(card.attack for card in live_enemies)
            return [card for card in live_enemies if card.attack == min_attack]


class SeabreakerGoliath(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 6
    base_health = 7
    base_windfury = True

    def overkill(self, context: CombatPhaseContext):
        bonus = 4 if self.golden else 2
        pirates = [card for card in context.friendly_war_party.board if card.check_type(MONSTER_TYPES.PIRATE) and card != self]
        for pirate in pirates:
            pirate.attack += bonus
            pirate.health += bonus


class FoeReaper4000(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.MECH
    base_attack = 6
    base_health = 9
    base_cleave = True  # TODO: need to implement cleave as an on-damage effect
    legendary = True

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == EVENTS.ON_ATTACK and event.card == self:
            live_enemy_board = [card for card in context.enemy_war_party.board if not card.dead]
            foe_index = live_enemy_board.index(event.foe)
            splash_damage_targets = [card for i, card in enumerate(live_enemy_board) if abs(i - foe_index) == 1]
            for target in splash_damage_targets:
                target.take_damage(self.attack, context.enemy_context(), self)
                target.resolve_death(context.enemy_context(), self)


class Amalgadon(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.ALL
    base_attack = 6
    base_health = 6

    def base_battlecry(self, targets: List['MonsterCard'], context: 'BuyPhaseContext'):
        count = len(one_minion_per_type(context.owner.in_play, context.randomizer)) * (2 if self.golden else 1)
        for _ in range(count):
            available_adaptations = valid_adaptations(self)
            adaptation = context.randomizer.select_adaptation(available_adaptations)
            self.adapt(adaptation())


class AnnoyOModule(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.MECH
    base_attack = 2
    base_health = 4
    base_divine_shield = True
    base_taunt = True
    base_magnetic = True


class Siegebreaker(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 5
    base_health = 8
    base_taunt = True

    def handle_event_powers(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.COMBAT_START or (event.event is EVENTS.SUMMON_COMBAT and event.card == self):
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


class CobaltScalebane(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 5
    base_health = 5

    def handle_event_powers(self, event: 'CardEvent', context: 'BuyPhaseContext'):
        if event.event is EVENTS.BUY_END:
            other_minions = [card for card in context.owner.in_play if card != self]
            if other_minions:
                card = context.randomizer.select_friendly_minion(other_minions)
                bonus = 6 if self.golden else 3
                card.attack += bonus


class TheBeast(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 9
    base_health = 7
    legendary = True

    def base_deathrattle(self, context: 'CombatPhaseContext'):
        for _ in range(context.summon_minion_multiplier()):
            context.enemy_war_party.summon_in_combat(FinkleEinhorn(), context.enemy_context())


class FinkleEinhorn(MonsterCard):
    tier = 1
    monster_type = None
    base_attack = 3
    base_health = 3
    token = True


class SouthseaStrongarm(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 5
    base_health = 4
    num_battlecry_targets = [1]

    def base_battlecry(self, targets: List['MonsterCard'], context: 'BuyPhaseContext'):
        if targets:
            num_purchased_pirates = len([card_type for card_type in context.owner.purchased_minions if
                                         card_type.check_type(MONSTER_TYPES.PIRATE)])
            bonus = num_purchased_pirates * (2 if self.golden else 1)
            targets[0].attack += bonus
            targets[0].health += bonus

    def valid_battlecry_target(self, card: 'MonsterCard') -> bool:
        return card.check_type(MONSTER_TYPES.PIRATE)


class CaveHydra(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 2
    base_health = 4
    base_cleave = True  # TODO: need to implement cleave as an on-damage effect

    # TODO: combat test


class PrimalfinLookout(MonsterCard):
    tier = 4
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 3
    base_health = 2

    def base_battlecry(self, targets: List['MonsterCard'], context: 'BuyPhaseContext'):
        murloc_in_play = [card for card in context.owner.in_play if card.check_type(MONSTER_TYPES.MURLOC)]
        if murloc_in_play:
            num_discovers = 2 if self.golden else 1
            for _ in range(num_discovers):
                context.owner.draw_discover(lambda card: card.check_type(MONSTER_TYPES.MURLOC) and card.tier <= context.owner.tavern_tier and type(card) != type(self))


class Murozond(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 5
    base_health = 5

    def base_battlecry(self, targets: List['MonsterCard'], context: 'BuyPhaseContext'):
        if context.owner.last_opponent_warband:
            card = context.randomizer.select_gain_card(context.owner.last_opponent_warband)
            plain_copy = type(card)()
            if self.golden:
                plain_copy.golden_transformation([])
            context.owner.gain_hand_card(plain_copy)


class PartyElemental(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 2
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.SUMMON_BUY and event.card.check_type(MONSTER_TYPES.ELEMENTAL) and event.card != self:
            other_elementals = [card for card in context.owner.in_play if card.check_type(MONSTER_TYPES.ELEMENTAL) and card != self]
            num_buffs = 2 if self.golden else 1
            for _ in range(num_buffs):
                random_elemental = context.randomizer.select_friendly_minion(other_elementals)
                random_elemental.attack += 1
                random_elemental.health += 1


class MoltenRock(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 2
    base_health = 3
    base_taunt = True

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.SUMMON_BUY and event.card.check_type(MONSTER_TYPES.ELEMENTAL) and event.card != self:
            self.health += 2 if self.golden else 1


class ArcaneAssistant(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 3
    base_health = 2

    def base_battlecry(self, targets: List['MonsterCard'], context: 'BuyPhaseContext'):
        bonus = 2 if self.golden else 1
        for card in context.owner.in_play:
            if card.check_type(MONSTER_TYPES.ELEMENTAL) and card != self:
                card.attack += bonus
                card.health += bonus


class CracklingCyclone(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 4
    base_health = 1
    base_divine_shield = True
    base_windfury = True

    def golden_transformation(self, base_cards: List['MonsterCard']):
        super().golden_transformation(base_cards)
        self.windfury = False
        self.mega_windfury = True


class DeadlySpore(MonsterCard):
    tier = 4
    monster_type = None
    base_attack = 1
    base_health = 1
    base_poisonous = True


class LieutenantGarr(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 8
    base_health = 1
    base_taunt = True

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.SUMMON_BUY and event.card.check_type(MONSTER_TYPES.ELEMENTAL) and event.card != self:
            num_elementals = len([card for card in context.owner.in_play if card.check_type(MONSTER_TYPES.ELEMENTAL)])
            multiplier = 2 if self.golden else 1
            self.health += num_elementals * multiplier


class Sellemental(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 2
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.SELL and event.card == self:
            num_tokens = 2 if self.golden else 1
            for _ in range(num_tokens):
                water_droplet = WaterDroplet()
                context.owner.gain_hand_card(water_droplet)


class WaterDroplet(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 2
    base_health = 2
    token = True


class LilRag(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 4
    base_health = 4
    legendary = True

    def handle_event_powers(self, event: 'CardEvent', context: Union['BuyPhaseContext', CombatPhaseContext]):
        if event.event is EVENTS.SUMMON_BUY and event.card.check_type(MONSTER_TYPES.ELEMENTAL) and event.card != self:
            num_buffs = 2 if self.golden else 1
            for _ in range(num_buffs):
                friendly_minion = context.randomizer.select_friendly_minion(context.owner.in_play)
                friendly_minion.attack += event.card.tier
                friendly_minion.health += event.card.tier


class TavernTempest(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 4
    base_health = 4

    def base_battlecry(self, targets: List['MonsterCard'], context: 'BuyPhaseContext'):
        for _ in range(2 if self.golden else 1):
            if context.owner.room_in_hand():
                available_elementals = [card for card in context.owner.tavern.deck.unique_cards() if card.check_type(
                    MONSTER_TYPES.ELEMENTAL) and card.tier <= context.owner.tavern_tier and type(card) != type(self)]
                random_elementals = context.randomizer.select_gain_card(available_elementals)
                context.owner.tavern.deck.remove_card(random_elementals)
                context.owner.gain_hand_card(random_elementals)


class GentleDjinni(MonsterCard):
    tier = 6
    monster_type = MONSTER_TYPES.ELEMENTAL
    base_attack = 6
    base_health = 8
    base_taunt = True
    legendary = True

    def base_deathrattle(self, context: 'CombatPhaseContext'):  # TODO: how does this minion interact with the pool?
        count = 2 if self.golden else 1
        summon_index = context.friendly_war_party.get_index(self)
        i = 0
        for _ in range(count):
            for _ in range(context.summon_minion_multiplier()):
                if context.friendly_war_party.room_on_board():
                    elementals = [card_type for card_type in PrintingPress.all_types() if
                                  card_type.check_type(MONSTER_TYPES.ELEMENTAL) and card_type != self]
                    random_elemental_type = context.randomizer.select_summon_minion(elementals)
                    context.friendly_war_party.summon_in_combat(random_elemental_type(), context, summon_index + i + 1)
                    i += 1
                    if context.friendly_war_party.owner.room_in_hand():
                        same_elemental_in_tavern = [card for card in
                                                    context.friendly_war_party.owner.tavern.deck.unique_cards() if
                                                    type(card) == random_elemental_type]
                        if same_elemental_in_tavern:
                            context.friendly_war_party.owner.tavern.deck.remove_card(same_elemental_in_tavern[0])
                        context.friendly_war_party.owner.gain_hand_card(random_elemental_type())