from typing import Union, List

from hearthstone.cards import MonsterCard, CardEvent, PrintingPress
from hearthstone.events import EVENTS, BuyPhaseContext, CombatPhaseContext
from hearthstone.monster_types import MONSTER_TYPES


class ArcaneCannon(MonsterCard):
    tier = 2
    monster_type = None
    base_attack = 2
    base_health = 2
    cant_attack = True

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if event.event is EVENTS.AFTER_ATTACK:
            count = 2 if self.golden else 1
            for _ in range(count):
                if event.card in context.friendly_war_party.board:
                    if abs(context.friendly_war_party.board.index(self) - context.friendly_war_party.board.index(
                            event.card)) == 1:
                        possible_targets = [card for card in context.enemy_war_party.board if not card.is_dying()]
                        if possible_targets:
                            target = context.randomizer.select_enemy_minion(possible_targets)
                            target.take_damage(2, context, self)
                            target.resolve_death(
                                CombatPhaseContext(context.enemy_war_party, context.friendly_war_party,
                                                   context.randomizer), self)


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


class TreasureChest(MonsterCard):
    tier = 1
    monster_type = None
    base_attack = 0
    base_health = 2
    token = True

    def base_deathrattle(self, context: CombatPhaseContext):
        count = 2 if self.golden else 1
        summon_index = context.friendly_war_party.get_index(self)
        i = 0
        for _ in range(count):
            for _ in range(context.summon_minion_multiplier()):
                all_minions = [minion() for minion in PrintingPress.all_types()]
                random_minion = context.randomizer.select_summon_minion(all_minions)
                random_minion.golden_transformation([])
                context.friendly_war_party.summon_in_combat(random_minion, context, summon_index + i + 1)
                i += 1
