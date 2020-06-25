from typing import Union

from hearthstone import events, combat
from hearthstone.cards import MonsterCard, CardEvent
from hearthstone.events import SUMMON_BUY, BuyPhaseContext, CombatPhaseContext, SUMMON_COMBAT, ON_ATTACK, COMBAT_START, \
    SELL
from hearthstone.monster_types import BEAST, DEMON, MECH, PIRATE, DRAGON, MURLOC


class MamaBear(MonsterCard):
    #  wrong tier for testing actual tier is 6
    tier = 6
    monster_type = BEAST
    base_attack = 5
    base_health = 5

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == SUMMON_BUY and event.card.monster_type == BEAST:
            event.card.attack += 5
            event.card.health += 5


class ShifterZerus(MonsterCard):
    tier = 3
    base_attack = 1
    base_health = 1


class SneedsOldShredder(MonsterCard):
    tier = 5
    monster_type = MECH
    base_attack = 5
    base_health = 7

    def base_deathrattle(self, context):
        pass  # summon random legendary minion


class FreedealingGambler(MonsterCard):
    tier = 2
    monster_type = PIRATE
    base_attack = 3
    base_health = 3

    def base_deathrattle(self, context):
        pass


class DragonspawnLieutenant(MonsterCard):
    tier = 1
    monster_type = DRAGON
    base_attack = 2
    base_health = 3
    base_taunt = True


class RighteousProtector(MonsterCard):
    tier = 1
    monster_type = None
    base_attack = 1
    base_health = 1
    base_taunt = True
    base_divine_shield = True


class AlleyCat(MonsterCard):
    tier = 1
    monster_type = BEAST
    base_attack = 1
    base_health = 1

    def base_battlecry(self, context: 'BuyPhaseContext'):
        tabby_cat = TabbyCat()
        if self.golden:
            tabby_cat.golden_transformation([])
        context.owner.summon_from_void(tabby_cat)


class TabbyCat(MonsterCard):
    token = True
    tier = 1
    monster_type = BEAST
    base_attack = 1
    base_health = 1


class ScavengingHyena(MonsterCard):
    tier = 1
    monster_type = BEAST
    base_attack = 2
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == events.DIES and event.card.monster_type == BEAST and event.card in context.friendly_war_party.board:
            self.attack += 2
            self.health += 1


class FiendishServant(MonsterCard):
    tier = 1
    monster_type = DEMON
    base_attack = 2
    base_health = 1

    def base_deathrattle(self, context: CombatPhaseContext):
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
        if event.event == events.SUMMON_BUY and event.card.monster_type == DEMON:
            context.owner.health -= 1
            self.attack += 2
            self.health += 2


class MechaRoo(MonsterCard):
    tier = 1
    monster_type = MECH
    base_attack = 1
    base_health = 1

    def base_deathrattle(self, context: CombatPhaseContext):
        joebot = JoEBot()
        if self.golden:
            joebot.golden_transformation([])
        summon_index = context.friendly_war_party.get_index(self)
        context.friendly_war_party.summon_in_combat(joebot, context, summon_index + 1)


class JoEBot(MonsterCard):
    token = True
    tier = 1
    monster_type = MECH
    base_attack = 1
    base_health = 1


class MicroMachine(MonsterCard):
    tier = 1
    monster_type = MECH
    base_attack = 1
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == events.BUY_START:
            if self.golden:
                self.attack += 2
            else:
                self.attack += 1


class MurlocTidecaller(MonsterCard):
    tier = 1
    monster_type = MURLOC
    base_attack = 1
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        friendly_summon = event.event == SUMMON_BUY or (
                event.event == SUMMON_COMBAT and event.card in context.friendly_war_party.board)
        if friendly_summon and event.card.monster_type == MURLOC and event.card != self:
            self.attack += bonus


class MurlocTidehunter(MonsterCard):
    tier = 1
    monster_type = MURLOC
    base_attack = 2
    base_health = 1

    def base_battlecry(self, context: BuyPhaseContext):
        murloc_scout = MurlocScout()
        if self.golden:
            murloc_scout.golden_transformation([])
        context.owner.summon_from_void(murloc_scout)


class MurlocScout(MonsterCard):
    token = True
    tier = 1
    monster_type = MURLOC
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
    monster_type = DEMON
    base_attack = 2
    base_health = 4
    base_taunt = True

    def base_battlecry(self, context: BuyPhaseContext):
        context.owner.health -= 2


class RedWhelp(MonsterCard):
    tier = 1
    monster_type = DRAGON
    base_attack = 1
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == events.COMBAT_START:
            num_friendly_dragons = len(
                [card for card in context.friendly_war_party.board if
                 not card.dead and card.monster_type == DRAGON])  # Red Whelp counts all dragons including itself
            targets = [card for card in context.enemy_war_party.board if not card.dead]
            if not targets:
                return
            num_damage_instances = 2 if self.golden else 1
            for _ in range(num_damage_instances):
                target = context.randomizer.select_enemy_minion(targets)
                target.take_damage(num_friendly_dragons)
                target.resolve_death(context)  # TODO: Order of death resolution?


class HarvestGolem(MonsterCard):
    tier = 2
    monster_type = MECH
    base_attack = 2
    base_health = 3

    def base_deathrattle(self, context: CombatPhaseContext):
        damaged_golem = DamagedGolem()
        if self.golden:
            damaged_golem.golden_transformation([])
        summon_index = context.friendly_war_party.get_index(self)
        context.friendly_war_party.summon_in_combat(damaged_golem, context, summon_index + 1)


class DamagedGolem(MonsterCard):
    token = True
    tier = 1
    monster_type = MECH
    base_attack = 2
    base_health = 1


class KaboomBot(MonsterCard):
    tier = 2
    monster_type = MECH
    base_attack = 2
    base_health = 2

    def base_deathrattle(self, context: CombatPhaseContext):
        num_damage_instances = 2 if self.golden else 1
        for _ in range(num_damage_instances):
            targets = [card for card in context.enemy_war_party.board if not card.dead]
            if not targets:
                break
            target = context.randomizer.select_enemy_minion(targets)
            target.take_damage(4)
            target.resolve_death(context)  # TODO: Order of death resolution?


class KindlyGrandmother(MonsterCard):
    tier = 2
    monster_type = BEAST
    base_attack = 1
    base_health = 1

    def base_deathrattle(self, context: CombatPhaseContext):
        big_bad_wolf = BigBadWolf()
        if self.golden:
            big_bad_wolf.golden_transformation([])
        context.friendly_war_party.summon_in_combat(big_bad_wolf, context)


class BigBadWolf(MonsterCard):
    token = True
    tier = 1
    monster_type = BEAST
    base_attack = 3
    base_health = 2


class MetaltoothLeaper(MonsterCard):
    tier = 2
    monster_type = BEAST
    base_attack = 3
    base_health = 3

    def base_battlecry(self, context: BuyPhaseContext):
        for card in context.owner.in_play:
            if card != self and card.monster_type == MECH:
                card.attack += 2


class RabidSaurolisk(MonsterCard):
    tier = 2
    monster_type = BEAST
    base_attack = 3
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == SUMMON_BUY and event.card.deathrattles:
            self.attack += 1
            self.health += 1


class GlyphGuardian(MonsterCard):
    tier = 2
    monster_type = DRAGON
    base_attack = 2
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event == ON_ATTACK and event.card == self:
            multiplier = 2
            if self.golden:
                multiplier = 3
            self.attack *= multiplier


class Imprisoner(MonsterCard):
    tier = 2
    monster_type = DEMON
    base_attack = 3
    base_health = 3

    def base_deathrattle(self, context: CombatPhaseContext):
        imp = Imp()
        if self.golden:
            imp.golden_transformation([])
        context.friendly_war_party.summon_in_combat(imp, context)


class Imp(MonsterCard):
    token = True
    tier = 1
    monster_type = DEMON
    base_attack = 1
    base_health = 1


class MurlocWarleader(MonsterCard):
    tier = 2
    monster_type = MURLOC
    base_attack = 3
    base_health = 3

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2
        if self.golden:
            bonus = 4
        if event.event == COMBAT_START or (event.event == SUMMON_COMBAT and event.card == self):
            murlocs = [card for card in context.friendly_war_party.board if
                       card != self and card.monster_type == MURLOC]
            for murloc in murlocs:
                murloc.attack += bonus
        elif event.event == SUMMON_COMBAT and event.card in context.friendly_war_party.board \
                and event.card != self and event.card.monster_type == MURLOC:
            event.card.attack += bonus

    def base_deathrattle(self, context: CombatPhaseContext):
        # TODO: IS THIS NEEDED?  Cause we have no idea...
        bonus = 2
        if self.golden:
            bonus = 4
        murlocs = [card for card in context.friendly_war_party.board if
                   card != self and card.monster_type == MURLOC]
        for murloc in murlocs:
            murloc.attack -= bonus


class StewardOfTime(MonsterCard):
    tier = 2
    monster_type = DRAGON
    base_attack = 3
    base_health = 4

    def handle_event_in_hand(self, event: CardEvent, context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        if event.event == SELL and event.card == self:
            for card in context.owner.store:
                card.attack += bonus
                card.health += bonus

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        self.handle_event_in_hand(event, context)


class Scallywag(MonsterCard):
    tier = 1
    monster_type = PIRATE
    base_attack = 2
    base_health = 1

    def base_deathrattle(self, context: CombatPhaseContext):
        pirate_summon = SkyPirate()
        if self.golden:
            pirate_summon.golden_transformation()
        scallywag_index = context.friendly_war_party.get_index(self)
        context.friendly_war_party.summon_in_combat(pirate_summon, context, scallywag_index + 1)


class SkyPirate(MonsterCard):
    tier = 1
    token = True
    monster_type = PIRATE
    base_attack = 1
    base_health = 1

    def handle_event_powers(self, event: CardEvent, context: CombatPhaseContext):
        if event.event == SUMMON_COMBAT and event.card == self:
            attacking_war_party = context.friendly_war_party
            defending_war_party = context.enemy_war_party
            attacker = self
            defender = defending_war_party.get_random_monster(context.randomizer)
            if not defender:
                return
            print(f'{attacking_war_party.owner.name} is attacking {defending_war_party.owner.name}')
            combat.start_attack(attacker, defender, attacking_war_party, defending_war_party, context.randomizer)
