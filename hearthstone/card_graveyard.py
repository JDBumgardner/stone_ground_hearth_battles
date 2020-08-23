from typing import Union, List

from hearthstone.cards import MonsterCard, CardEvent
from hearthstone.events import EVENTS, BuyPhaseContext, CombatPhaseContext
from hearthstone.monster_types import MONSTER_TYPES


class ArcaneCannon(MonsterCard):  # TODO: Removed in latest patch... how should we deal with this?
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


class Zoobot(MonsterCard):  # TODO: This was removed in a previous patch... how should we deal with this?
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