import logging
from typing import Union, List

from hearthstone import combat
from hearthstone.cards import MonsterCard, CardEvent
from hearthstone.events import BuyPhaseContext, CombatPhaseContext, EVENTS
from hearthstone.monster_types import MONSTER_TYPES

class MamaBear(MonsterCard):
    #  wrong tier for testing actual tier is 6
    tier = 6
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 5
    base_health = 5

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.SUMMON_BUY and event.card.monster_type == MONSTER_TYPES.BEAST:
            event.card.attack += 5
            event.card.health += 5


class ShifterZerus(MonsterCard):
    tier = 3
    base_attack = 1
    base_health = 1


class SneedsOldShredder(MonsterCard):
    tier = 5
    monster_type = MONSTER_TYPES.MECH
    base_attack = 5
    base_health = 7

    def base_deathrattle(self, context):
        pass  # summon random legendary minion


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

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.DIES and event.card.monster_type == MONSTER_TYPES.BEAST and event.card in context.friendly_war_party.board:
            self.attack += 2
            self.health += 1


class FiendishServant(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.DEMON
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
        if event.event is EVENTS.SUMMON_BUY and event.card.monster_type == MONSTER_TYPES.DEMON:
            bonus = 4 if self.golden else 2
            context.owner.health -= 1
            self.attack += bonus
            self.health += bonus


class MechaRoo(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MECH
    base_attack = 1
    base_health = 1

    def base_deathrattle(self, context: CombatPhaseContext):
        summon_index = context.friendly_war_party.get_index(self)
        for i in range(context.summon_minion_multiplier()):
            joebot = JoEBot()
            if self.golden:
                joebot.golden_transformation([])
            context.friendly_war_party.summon_in_combat(joebot, context, summon_index + i + 1)


class JoEBot(MonsterCard):
    token = True
    tier = 1
    monster_type = MONSTER_TYPES.MECH
    base_attack = 1
    base_health = 1


class MicroMachine(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MECH
    base_attack = 1
    base_health = 2

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
        if friendly_summon and event.card.monster_type == MONSTER_TYPES.MURLOC and event.card != self:
            self.attack += bonus


class MurlocTidehunter(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 2
    base_health = 1

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

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        context.owner.health -= 2


class RedWhelp(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 1
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.COMBAT_START:
            num_friendly_dragons = len(
                [card for card in context.friendly_war_party.board if
                 not card.dead and card.monster_type == MONSTER_TYPES.DRAGON])  # Red Whelp counts all dragons including itself
            targets = [card for card in context.enemy_war_party.board if not card.dead]
            if not targets:
                return
            num_damage_instances = 2 if self.golden else 1
            for _ in range(num_damage_instances):
                target = context.randomizer.select_enemy_minion(targets)
                target.take_damage(num_friendly_dragons, context)
                target.resolve_death(context)  # TODO: Order of death resolution?


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
            targets = [card for card in context.enemy_war_party.board if not card.dead]
            if not targets:
                break
            target = context.randomizer.select_enemy_minion(targets)
            target.take_damage(4, context)
            target.resolve_death(context)  # TODO: Order of death resolution?


class KindlyGrandmother(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 1
    base_health = 1

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
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 3
    base_health = 3

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        for card in context.owner.in_play:
            if card != self and card.monster_type == MONSTER_TYPES.MECH:
                card.attack += 2


class RabidSaurolisk(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.BEAST
    base_attack = 3
    base_health = 2

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.SUMMON_BUY and event.card.deathrattles:
            self.attack += 1
            self.health += 1


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
        bonus = 2
        if self.golden:
            bonus = 4
        if event.event is EVENTS.COMBAT_START or (event.event is EVENTS.SUMMON_COMBAT and event.card == self):
            murlocs = [card for card in context.friendly_war_party.board if
                       card != self and card.monster_type == MONSTER_TYPES.MURLOC]
            for murloc in murlocs:
                murloc.attack += bonus
        elif event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board \
                and event.card != self and event.card.monster_type == MONSTER_TYPES.MURLOC:
            event.card.attack += bonus

    def base_deathrattle(self, context: CombatPhaseContext):
        # TODO: IS THIS NEEDED?  Cause we have no idea... Jarett
        bonus = 2
        if self.golden:
            bonus = 4
        murlocs = [card for card in context.friendly_war_party.board if
                   card != self and card.monster_type == MONSTER_TYPES.MURLOC]
        for murloc in murlocs:
            murloc.attack -= bonus


class StewardOfTime(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.DRAGON
    base_attack = 3
    base_health = 4

    def handle_event_in_hand(self, event: CardEvent, context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.SELL and event.card == self:
            for card in context.owner.store:
                card.attack += bonus
                card.health += bonus

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        self.handle_event_in_hand(event, context)


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
            scallywag_index = context.friendly_war_party.get_index(self)
            context.friendly_war_party.summon_in_combat(pirate_summon, context, scallywag_index + i + 1)


class SkyPirate(MonsterCard):
    tier = 1
    token = True
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 1
    base_health = 1

    def handle_event_powers(self, event: CardEvent, context: CombatPhaseContext):
        if event.event is EVENTS.SUMMON_COMBAT and event.card == self:
            attacking_war_party = context.friendly_war_party
            defending_war_party = context.enemy_war_party
            attacker = self
            defender = defending_war_party.get_random_monster(context.randomizer)
            if not defender:
                return
            logging.debug(f'{attacking_war_party.owner.name} is attacking {defending_war_party.owner.name}')
            combat.start_attack(attacker, defender, attacking_war_party, defending_war_party, context.randomizer)


class DeckSwabbie(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 2
    base_health = 2

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        discount = 2 if self.golden else 1
        context.owner.tavern_upgrade_cost = max(context.owner.tavern_upgrade_cost - discount, 0)


class UnstableGhoul(MonsterCard):
    tier = 2
    monster_type = None
    base_attack = 1
    base_health = 3
    base_taunt = True

    def base_deathrattle(self, context: CombatPhaseContext):
        all_minions = [card for card in context.friendly_war_party.board + context.enemy_war_party.board if
                       not card.dead]

        count = 2 if self.golden else 1
        for _ in range(count):
            for minion in all_minions:
                if minion.dead:
                    continue
                minion.take_damage(1, context)
                minion.resolve_death(context)  # TODO: Order of death resolution?


class RockpoolHunter(MonsterCard):
    tier = 1
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 2
    base_health = 3
    num_battlecry_targets = 1

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def validate_battlecry_target(self, card: MonsterCard) -> bool:
        return card.monster_type == MONSTER_TYPES.MURLOC and card != self


class RatPack(MonsterCard):
    tier = 2
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


class ArcaneCannon(MonsterCard):
    tier = 2
    monster_type = None
    base_attack = 2
    base_health = 2
    cant_attack = True

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        damage = 4 if self.golden else 2
        if event.event is EVENTS.ON_ATTACK:
            friendly_live_war_party = [friend for friend in context.friendly_war_party.board if not friend.dead]
            if event.card in friendly_live_war_party:
                if abs(friendly_live_war_party.index(self) - friendly_live_war_party.index(event.card)) == 1:
                    target = context.randomizer.select_enemy_minion([card for card in context.enemy_war_party.board if card])
                    target.take_damage(damage, context)
                    target.resolve_death(CombatPhaseContext(context.enemy_war_party, context.friendly_war_party, context.randomizer))


class NathrezimOverseer(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.DEMON
    base_attack = 2
    base_health = 3

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def validate_battlecry_target(self, card: MonsterCard) -> bool:
        return card.monster_type == MONSTER_TYPES.DEMON and card != self


class OldMurkeye(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 2
    base_health = 4

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 2 if self.golden else 1
        if event.event is EVENTS.COMBAT_START:
            self.attack += bonus * sum(1 for murloc in context.friendly_war_party.board if murloc.monster_type is MONSTER_TYPES.MURLOC)
        if event.event is EVENTS.DIES and event.card in context.friendly_war_party.board and event.card.monster_type is MONSTER_TYPES.MURLOC:
            self.attack -= bonus
        if event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board and event.card.monster_type is MONSTER_TYPES.MURLOC:
            self.attack += bonus


class CrystalWeaver(MonsterCard):
    tier = 3
    base_attack = 5
    base_health = 4

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        for card in context.owner.in_play:
            if card.monster_type == MONSTER_TYPES.DEMON:
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


class Zoobot(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.MECH
    base_attack = 3
    base_health = 3

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        for monster_type in [MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.DRAGON]:
            friendly_cards = [card for card in context.owner.in_play if card.monster_type == monster_type]
            if friendly_cards:
                card = context.randomizer.select_friendly_minion(friendly_cards)
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
            if card.monster_type == MONSTER_TYPES.PIRATE and card != self:
                card.attack += bonus


class ColdlightSeer(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 2
    base_health = 3

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        for card in context.owner.in_play:
            if card.monster_type == MONSTER_TYPES.MURLOC and card != self:
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
        if event.event is EVENTS.SUMMON_COMBAT and event.card.monster_type == MONSTER_TYPES.MECH and event.card in context.friendly_war_party.board:
            self.attack += bonus
            self.divine_shield = True


class FelfinNavigator(MonsterCard):
    tier = 3
    monster_type = MONSTER_TYPES.MURLOC
    base_attack = 4
    base_health = 4

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 2 if self.golden else 1
        for card in context.owner.in_play:
            if card.monster_type == MONSTER_TYPES.MURLOC and card != self:
                card.health += bonus
                card.attack += bonus


class Houndmaster(MonsterCard):
    tier = 3
    base_attack = 4
    base_health = 3
    num_battlecry_targets = 1

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus
            targets[0].taunt = True

    def validate_battlecry_target(self, card: MonsterCard) -> bool:
        return card.monster_type == MONSTER_TYPES.BEAST


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
        for i in range(2*context.summon_minion_multiplier()):
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
    base_attack = 3
    base_health = 2

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.AFTER_ATTACK and self == event.card:
            friendly_deathrattlers = [card for card in context.friendly_war_party.board if card != self and not card.dead
                                      and card.deathrattles]
            if friendly_deathrattlers:
                deathrattle_triggers = 2 if self.golden else 1
                for _ in range(deathrattle_triggers):
                    deathrattler = context.randomizer.select_friendly_minion(friendly_deathrattlers)
                    deathrattler.base_deathrattle(context)


class ScrewjankClunker(MonsterCard):
    tier = 3
    base_attack = 2
    base_health = 5
    monster_type = MONSTER_TYPES.MECH
    num_battlecry_targets = 1

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def validate_battlecry_target(self, card: MonsterCard) -> bool:
        return card.monster_type == MONSTER_TYPES.MECH


class PackLeader(MonsterCard):
    tier = 3
    base_attack = 3
    base_health = 3
    monster_type = None

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        friendly_summon = event.event is EVENTS.SUMMON_BUY or (
                event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board)
        if friendly_summon and event.card.monster_type == MONSTER_TYPES.BEAST and event.card != self:
            bonus = 6 if self.golden else 3
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
                two_cost_minions = [VulgarHomunculus(), MicroMachine(), MurlocTidehunter(), RockpoolHunter(),
                                    DragonspawnLieutenant(), KindlyGrandmother(), ScavengingHyena(), UnstableGhoul(),
                                    Khadgar()]
                random_minion = context.randomizer.select_summon_minion(two_cost_minions)
                context.friendly_war_party.summon_in_combat(random_minion, context, summon_index + i + 1)
                i += 1


class SaltyLooter(MonsterCard):
    tier = 3
    base_attack = 3
    base_health = 3
    monster_type = MONSTER_TYPES.PIRATE

    def handle_event_powers(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.SUMMON_BUY and event.card.monster_type == MONSTER_TYPES.PIRATE and event.card != self:
            bonus = 2 if self.golden else 1
            self.attack += bonus
            self.health += bonus


class SoulJuggler(MonsterCard):
    tier = 3
    base_attack = 3
    base_health = 3
    monster_type = None

    def handle_event_powers(self, event: CardEvent, context: CombatPhaseContext):
        if event.event is EVENTS.DIES and event.card.monster_type == MONSTER_TYPES.DEMON and event.card in context.friendly_war_party.board:
            count = 2 if self.golden else 1
            for _ in range(count):
                targets = [card for card in context.enemy_war_party.board if not card.dead and not card.health <= 0]
                if targets:
                    target = context.randomizer.select_enemy_minion(targets)
                    target.take_damage(3, context)
                    target.resolve_death(context)  # TODO: Order of death resolution?


class TwilightEmissary(MonsterCard):
    tier = 3
    base_attack = 4
    base_health = 4
    monster_type = MONSTER_TYPES.DRAGON
    base_taunt = True
    num_battlecry_targets = 1

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def validate_battlecry_target(self, card: MonsterCard) -> bool:
        return card.monster_type == MONSTER_TYPES.DRAGON


class Khadgar(MonsterCard):
    tier = 3
    base_attack = 2
    base_health = 2
    monster_type = None

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
    num_battlecry_targets = 1

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        bonus = 4 if self.golden else 2
        if targets:
            targets[0].attack += bonus
            targets[0].health += bonus

    def validate_battlecry_target(self, card: MonsterCard) -> bool:
        return card.monster_type == MONSTER_TYPES.BEAST


class RipsnarlCaptain(MonsterCard):
    tier = 4
    base_attack = 3
    base_health = 4
    monster_type = MONSTER_TYPES.PIRATE

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.ON_ATTACK and event.card.monster_type == MONSTER_TYPES.PIRATE and event.card in context.friendly_war_party.board and event.card != self:
            bonus = 4 if self.golden else 2
            event.card.attack += bonus
            event.card.health += bonus


class DefenderOfArgus(MonsterCard):
    tier = 4
    base_attack = 2
    base_health = 3
    monster_type = None

    def base_battlecry(self, targets: List[MonsterCard], context: BuyPhaseContext):
        index = context.owner.in_play.index(self)
        bonus = 2 if self.golden else 1
        for i in [index-1, index+1]:
            try:
                adjacent_minion = context.owner.in_play[i]
            except IndexError:
                continue
            adjacent_minion.attack += bonus
            adjacent_minion.health += bonus
            adjacent_minion.taunt = True


class SouthseaCaptain(MonsterCard):
    tier = 2
    monster_type = MONSTER_TYPES.PIRATE
    base_attack = 3
    base_health = 3

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        bonus = 1
        if self.golden:
            bonus = 2
        if event.event is EVENTS.COMBAT_START or (event.event is EVENTS.SUMMON_COMBAT and event.card == self):
            pirates = [card for card in context.friendly_war_party.board if
                       card != self and card.monster_type == MONSTER_TYPES.PIRATE]
            for pirate in pirates:
                pirate.attack += bonus
                pirate.health += bonus
        elif event.event is EVENTS.SUMMON_COMBAT and event.card in context.friendly_war_party.board \
                and event.card != self and event.card.monster_type == MONSTER_TYPES.PIRATE:
            event.card.attack += bonus
            event.card.health += bonus

    def base_deathrattle(self, context: CombatPhaseContext):
        bonus = 1
        if self.golden:
            bonus = 2
        pirates = [card for card in context.friendly_war_party.board if
                   card != self and card.monster_type == MONSTER_TYPES.PIRATE]
        for pirate in pirates:
            pirate.attack -= bonus
            if pirate.health > pirate.base_health > pirate.health - bonus:
                pirate.health = pirate.base_health
