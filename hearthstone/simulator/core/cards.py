import copy
import enum
import itertools
import typing
from collections import defaultdict
from typing import List, Optional, Callable, Type, Union, Iterator

from boltons.setutils import IndexedSet

from hearthstone.simulator.core import events
from hearthstone.simulator.core.events import BuyPhaseContext, CombatPhaseContext, EVENTS, CardEvent
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.randomizer import Randomizer

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.adaptations import Adaptation
    from hearthstone.simulator.core.player import Player


def one_minion_per_type(cards: List['MonsterCard'], randomizer: 'Randomizer',
                        excluded_card: Optional['MonsterCard'] = None) -> List['MonsterCard']:
    minions = []
    restricted_cards = [card for card in cards]
    if excluded_card in restricted_cards:
        restricted_cards.remove(excluded_card)
    filler_minions = [card for card in restricted_cards if card.monster_type == MONSTER_TYPES.ALL]
    for minion_type in MONSTER_TYPES.single_types():
        minions_by_type = [card for card in restricted_cards if card.monster_type == minion_type]
        if minions_by_type:
            card = randomizer.select_friendly_minion(minions_by_type)
            minions.append(card)
        elif filler_minions:
            card = randomizer.select_friendly_minion(filler_minions)
            filler_minions.remove(card)
            minions.append(card)
    return minions


BOOL_ATTRIBUTE_LIST = ["divine_shield", "magnetic", "poisonous", "taunt", "windfury", "cleave", "reborn",
                       "mega_windfury", "gruul_rules", "frenzy"]


class MonsterCard:
    coin_cost = 3
    mana_cost: Optional[int] = None
    base_health: int
    base_attack: int
    monster_type: 'MONSTER_TYPES' = MONSTER_TYPES.NEUTRAL
    base_divine_shield = False
    base_magnetic = False
    base_poisonous = False
    base_taunt = False
    base_windfury = False
    base_cleave = False
    base_deathrattle = None
    base_battlecry = None
    num_battlecry_targets = [0]
    base_reborn = False
    redeem_rate = 1
    tier: int
    not_in_pool = False
    cant_attack = False
    give_immunity = False
    legendary = False
    pool: 'MONSTER_TYPES' = MONSTER_TYPES.ALL
    divert_taunt_attack = False

    def __init__(self):
        self.health = self.base_health
        self.attack = self.base_attack
        self.divine_shield = self.base_divine_shield
        self.magnetic = self.base_magnetic
        self.poisonous = self.base_poisonous
        self.taunt = self.base_taunt
        self.windfury = self.base_windfury
        self.mega_windfury = False
        self.cleave = self.base_cleave
        self.deathrattles: List[Callable[[MonsterCard, CombatPhaseContext], None]] = []
        if self.base_deathrattle is not None:
            self.deathrattles.append(self.base_deathrattle.__func__)
        self.reborn = self.base_reborn
        self.dead = False
        self.golden = False
        self.battlecry: Optional[Callable[[List[MonsterCard], CombatPhaseContext], None]] = self.base_battlecry
        self.attached_cards = []
        self.frozen = False
        self.nomi_buff = 0
        self.ticket = False
        self.token = self.not_in_pool
        self.link: Optional['MonsterCard'] = None  # links a card during combat to itself in the buy phase board
        self.dealt_lethal_damage_by = None
        self.frenzy_triggered = False
        self.gruul_rules = False

    def __repr__(self):
        rep = f"{type(self).__name__} {self.attack}/{self.health} (t{self.tier})"  # TODO: add a proper enum to the monster typing
        if self.dead:
            rep += ", [dead]"
        if self.battlecry:
            rep += ", [battlecry]"
        for attribute in BOOL_ATTRIBUTE_LIST:
            if getattr(self, attribute):
                rep += f", [{attribute}]"
        if self.deathrattles:
            rep += ", [%s]" % ",".join([f"deathrattle-{i}" for i in range(len(self.deathrattles))])
        if self.golden:
            rep += ", [golden]"
        if self.frozen:
            rep += ", [frozen]"
        if self.ticket:
            rep += ", [ticket]"

        return "{" + rep + "}"

    def take_damage(self, damage: int, combat_phase_context: CombatPhaseContext, foe: Optional['MonsterCard'] = None,
                    defending: Optional[bool] = True):
        if self.divine_shield and not damage <= 0:
            self.divine_shield = False
            combat_phase_context.broadcast_combat_event(events.DivineShieldLostEvent(self, foe=foe))
        else:
            self.health -= damage
            if foe is not None and foe.poisonous and self.health > 0:
                self.health = 0
            if defending and foe is not None and self.health < 0:
                foe.overkill(combat_phase_context.enemy_context())
            combat_phase_context.damaged_minions.add(self)
            combat_phase_context.broadcast_combat_event(events.CardDamagedEvent(self, foe=foe))
            if self.is_dying():
                self.dealt_lethal_damage_by = foe
            elif self.health >= 0 and not self.dead and not self.frenzy_triggered:
                self.frenzy(combat_phase_context)
                self.frenzy_triggered = True

    def resolve_death(self, context: CombatPhaseContext, foe: Optional['MonsterCard'] = None):
        if self.is_dying():
            self.dead = True
            context.friendly_war_party.dead_minions.append(self)
            context.event_queue.load_minion(EVENTS.DIES, context.friendly_war_party, self, foe)

    def trigger_reborn(self, context: CombatPhaseContext):
        index = context.friendly_war_party.get_index(self)
        for i in range(context.summon_minion_multiplier()):
            reborn_self = self.unbuffed_copy()
            reborn_self.health = 1
            reborn_self.reborn = False
            context.friendly_war_party.summon_in_combat(reborn_self, context, index + i + 1)

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if self == event.card:
            if event.event is EVENTS.DIES:
                if self.deathrattles:
                    context.event_queue.load_minion(EVENTS.DEATHRATTLE_TRIGGERED, context.friendly_war_party, self)
                if self.reborn:
                    self.trigger_reborn(context)
            elif event.event is EVENTS.SUMMON_BUY:
                if self.magnetic:
                    self.magnetize(event.targets, context)
                if self.battlecry:
                    for _ in range(context.battlecry_multiplier()):
                        self.battlecry(event.targets, context)
        if event.event is EVENTS.BUY_END:
            if self.gruul_rules:
                self.attack += 2
                self.health += 2
        if not self.dead or self == event.card:  # minions will trigger their own death events
            self.handle_event_powers(event, context)

    def handle_event_in_hand(self, event: CardEvent, context: BuyPhaseContext):
        return

    def handle_event_powers(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        return

    def valid_battlecry_target(self, card: 'MonsterCard') -> bool:
        return True

    def golden_transformation(self, base_cards: List['MonsterCard']):
        self.attack += self.base_attack
        self.health += self.base_health
        self.golden = True
        for card in base_cards:
            self.health += card.health - card.base_health
            self.attack += card.attack - card.base_attack
            self.attached_cards.extend(card.attached_cards)
            if card.base_deathrattle:
                self.deathrattles.extend(card.deathrattles[1:])
            else:
                self.deathrattles.extend(card.deathrattles)
            for attr in BOOL_ATTRIBUTE_LIST:
                if getattr(card, attr):
                    if attr == "windfury" and card.base_windfury:
                        setattr(self, "mega_windfury", True)
                        setattr(self, attr, False)
                    else:
                        setattr(self, attr, True)

    def magnetize(self, targets: List['MonsterCard'], context: 'BuyPhaseContext'):
        if targets:
            targets[0].attack += self.attack
            targets[0].health += self.health
            if self.deathrattles:
                targets[0].deathrattles.extend(self.deathrattles)
            for attr in BOOL_ATTRIBUTE_LIST:
                if getattr(self, attr) and attr != 'magnetic':
                    setattr(targets[0], attr, True)
            targets[0].attached_cards.append(self)
            context.owner.remove_board_card(self)

    def overkill(self, context: CombatPhaseContext):
        return

    def frenzy(self, context: CombatPhaseContext):
        return

    def dissolve(self) -> List['MonsterCard']:
        golden_modifier = 3 if self.golden else 1
        attached_cards = []
        for card in self.attached_cards:
            attached_cards.extend(card.dissolve())
        if self.token:
            return attached_cards
        else:
            dissolving_cards = [type(self)() for _ in range(golden_modifier)]
            dissolving_cards.extend(attached_cards)
            return dissolving_cards

    def summon_minion_multiplier(self) -> int:
        return 1

    def deathrattle_multiplier(self) -> int:
        return 1

    def battlecry_multiplier(self) -> int:
        return 1

    @classmethod
    def check_type(cls, desired_type: 'MONSTER_TYPES') -> bool:
        return cls.monster_type in (desired_type, MONSTER_TYPES.ALL)

    def is_targetable(self) -> bool:
        return not self.dead and self.health > 0

    def is_dying(self) -> bool:
        return not self.dead and self.health <= 0

    def adapt(self, adaptation: 'Adaptation'):
        assert adaptation.valid(self)
        adaptation.apply(self)

    def unbuffed_copy(self) -> 'MonsterCard':
        copy = type(self)()
        if self.golden:
            copy.golden_transformation([])
        return copy

    def valid_attack_targets(self, live_enemies: List['MonsterCard']) -> List['MonsterCard']:
        if self.attack <= 0 or not live_enemies:
            return []
        taunt_monsters = [card for card in live_enemies if card.taunt]
        if taunt_monsters:
            return taunt_monsters
        else:
            return live_enemies

    def num_attacks(self):
        if self.mega_windfury:
            return 4
        elif self.windfury:
            return 2
        else:
            return 1

    def apply_nomi_buff(self, player: 'Player'):
        if self.check_type(MONSTER_TYPES.ELEMENTAL):
            self.attack += (player.nomi_bonus - self.nomi_buff)
            self.health += (player.nomi_bonus - self.nomi_buff)
            self.nomi_buff = player.nomi_bonus

    def copy(self) -> 'MonsterCard':
        """
        For when gaining a "copy" of a card.
        :return:
        """
        clone = copy.copy(self)
        clone.deathrattles = clone.deathrattles.copy()
        clone.attached_cards = [card.copy() for card in clone.attached_cards]
        return clone

    def adjacent_minions(self, context: 'BuyPhaseContext', predicate: Callable[['MonsterCard'], bool]) -> List[
        'MonsterCard']:
        return [card for card in context.owner.in_play if
                predicate(card) and abs(context.owner.in_play.index(card) - context.owner.in_play.index(self)) == 1]


class CardList:
    def __init__(self, cards: List[MonsterCard]):
        # We use an IndexedSet instead of a set here for deterministic iteration order.
        self.cards_by_tier: typing.DefaultDict[int, IndexedSet] = defaultdict(lambda: IndexedSet())
        for card in cards:
            self.cards_by_tier[card.tier].add(card)

    def draw(self, player: 'Player', num: int) -> List['MonsterCard']:
        valid_cards = []
        for tier in range(player.tavern_tier + 1):
            valid_cards.extend(self.cards_by_tier[tier])

        selected_cards = []
        for i in range(num):
            assert valid_cards, "fnord"
            random_card = player.tavern.randomizer.select_draw_card(valid_cards, player.name, player.tavern.turn_count)
            self.cards_by_tier[random_card.tier].remove(random_card)
            valid_cards.remove(random_card)
            selected_cards.append(random_card)
        return selected_cards

    def draw_with_predicate(self, player: 'Player', predicate: Callable) -> 'MonsterCard':
        valid_cards = []
        for tier in range(player.tavern_tier + 1):
            valid_cards.extend([card for card in self.cards_by_tier[tier] if predicate(card)])
        assert valid_cards, "fnord"
        random_card = player.tavern.randomizer.select_draw_card(valid_cards, player.name, player.tavern.turn_count)
        self.cards_by_tier[random_card.tier].remove(random_card)
        return random_card

    def return_cards(self, cards: Iterator[MonsterCard]):
        for card in cards:
            self.return_card(card)

    def return_card(self, card: MonsterCard):
        self.cards_by_tier[card.tier].add(card)

    def remove_card(self, card: MonsterCard):
        self.cards_by_tier[card.tier].remove(card)

    def remove_card_of_type(self, card_type: 'Type'):
        cards_of_type = [card for card in self.cards_by_tier[card_type.tier] if type(card) == card_type]
        self.cards_by_tier[card_type.tier].remove(cards_of_type[0])

    def all_cards(self):
        return itertools.chain.from_iterable(self.cards_by_tier.values())

    def __len__(self) -> int:
        return sum(len(value) for value in self.cards_by_tier.values())

    def unique_cards(self) -> List['MonsterCard']:
        cards_by_type = {type(card): card for card in self.all_cards()}
        return list(cards_by_type.values())

    def cards_of_monstertype(self, monster_type: 'MONSTER_TYPES'):
        return [card for card in self.all_cards() if card.check_type(monster_type)]

    def cards_with_battlecry(self):
        return [card for card in self.all_cards() if card.base_battlecry]


class CardLocation(enum.Enum):
    STORE = 1
    HAND = 2
    BOARD = 3
    DISCOVER = 4
