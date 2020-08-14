import itertools
from collections import defaultdict
from typing import Set, List, Optional, Callable, Type, Union, Iterator

from hearthstone.events import BuyPhaseContext, CombatPhaseContext, EVENTS
from hearthstone.card_factory import make_metaclass


class PrintingPress:
    cards: Set[Type['Card']] = set()
    cards_per_tier = {1: 18, 2: 15, 3: 13, 4: 11, 5: 9, 6: 6}

    @classmethod
    def make_cards(cls) -> 'CardList':
        cardlist = []
        for card in cls.cards:
            if not card.token:
                cardlist.extend([card() for _ in range(cls.cards_per_tier[card.tier])])
        return CardList(cardlist)

    @classmethod
    def add_card(cls, card_class):
        cls.cards.add(card_class)


class CardEvent:
    def __init__(self, card: Optional['MonsterCard'], event: EVENTS, targets: Optional[List['MonsterCard']] = None):
        self.card = card
        self.event = EVENTS(event)
        self.targets = targets


CardType = make_metaclass(PrintingPress.add_card, ("Card", "MonsterCard"))


class Card(metaclass=CardType):
    type_name = "card"
    mana_cost: int
    card_name: str
    coin_cost = 3
    redeem_rate = 1
    tier: int
    token = False
    tracked = False

    def __init__(self):
        self.state = None
        self.tavern = None


class MonsterCard(Card):
    type_name = "monster"
    mana_cost = 0
    base_health: int
    base_attack: int
    monster_type = None
    base_divine_shield = False
    base_magnetic = False
    base_poisonous = False
    base_taunt = False
    base_windfury = False
    base_cleave = False
    base_deathrattle = None
    base_battlecry = None
    num_battlecry_targets = 0
    base_reborn = False
    token = False
    cant_attack = False
    shifting = False
    attached_cards = []

    def __init__(self):
        super().__init__()
        self.health = self.base_health
        self.attack = self.base_attack
        self.divine_shield = self.base_divine_shield
        self.magnetic = self.base_magnetic
        self.poisonous = self.base_poisonous
        self.taunt = self.base_taunt
        self.windfury = self.base_windfury
        self.cleave = self.base_cleave
        self.deathrattles: List[Callable[[CombatPhaseContext], None]] = []
        if self.base_deathrattle is not None:
            self.deathrattles.append(self.base_deathrattle.__func__)
        self.reborn = self.base_reborn
        self.dead = False
        self.golden = False
        self.battlecry: Optional[Callable[[CombatPhaseContext], None]] = self.base_battlecry
        self.bool_attribute_list = [
            "divine_shield", "magnetic", "poisonous", "taunt",
            "windfury", "cleave", "reborn"
        ]

    def __repr__(self):
        rep = f"{type(self).__name__} {self.attack}/{self.health} (t{self.tier})" #  TODO: add a proper enum to the monster typing
        if self.dead:
            rep += ", [dead]"
        if self.battlecry:
            rep += ", [battlecry]"
        for attribute in self.bool_attribute_list:
            if getattr(self, attribute):
                rep += f", [{attribute}]"
        if self.deathrattles:
            rep += ", [%s]" % ",".join([f"deathrattle-{i}" for i in range(len(self.deathrattles))])
        if self.golden:
            rep += ", [golden]"
        if self.shifting:
            rep += ", [shifting]"

        return "{" + rep + "}"

    def take_damage(self, damage: int, combat_phase_context: CombatPhaseContext, foe: Optional['MonsterCard'] = None, defending: Optional[bool] = True):
        if self.divine_shield and not damage <= 0:
            self.divine_shield = False
            combat_phase_context.broadcast_combat_event(CardEvent(self, EVENTS.DIVINE_SHIELD_LOST))
        else:
            self.health -= damage
            if foe is not None and foe.poisonous and self.health > 0:
                self.health = 0
            if defending and foe is not None and self.health < 0:
                foe.overkill(combat_phase_context)
            combat_phase_context.broadcast_combat_event(CardEvent(self, EVENTS.CARD_DAMAGED))

    def resolve_death(self, context: CombatPhaseContext):
        if self.health <= 0:
            self.dead = True
            card_death_event = CardEvent(self, EVENTS.DIES)
            context.broadcast_combat_event(card_death_event)

    def resolve_reborn(self, context: CombatPhaseContext):
        reborn_self = type(self)()
        if self.golden:
            reborn_self.golden_transformation([])
        reborn_self.health = 1
        reborn_self.reborn = False
        index = context.friendly_war_party.get_index(self)
        context.friendly_war_party.board.remove(self)
        context.friendly_war_party.board.insert(index, reborn_self)

    def change_state(self, new_state):
        self.tavern.run_callbacks(self, new_state)
        self.state = new_state

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if self == event.card:
            if event.event is EVENTS.DIES:
                for _ in range(context.deathrattle_multiplier()):
                    for deathrattle in self.deathrattles:
                        deathrattle(self, context)
                if self.reborn:
                    self.resolve_reborn(context)
            elif event.event is EVENTS.SUMMON_BUY:
                if self.magnetic:
                    self.magnetize(event.targets, context)
                if self.battlecry:
                    for _ in range(context.battlecry_multiplier()):
                        self.battlecry(event.targets, context)
                if event.card.tracked:
                    context.owner.counted_cards[type(event.card)] += 1
                self.shifting = False
        if not self.dead:
            self.handle_event_powers(event, context)

    def handle_event_in_hand(self, event: CardEvent, context: BuyPhaseContext):
        if event.event is EVENTS.BUY_START and self.shifting:
            self.zerus_shift(context)

    def handle_event_powers(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        return

    def validate_battlecry_target(self, card: 'MonsterCard') -> bool:
        return True

    def golden_transformation(self, base_cards: List['MonsterCard']):
        self.attack += self.base_attack
        self.health += self.base_health
        self.golden = True
        for card in base_cards:
            self.health += card.health - card.base_health
            self.attack += card.attack - card.base_attack
            if card.base_deathrattle:
                self.deathrattles.extend(card.deathrattles[1:])
            else:
                self.deathrattles.extend(card.deathrattles)
            for attr in card.bool_attribute_list:
                if getattr(card, attr):
                    setattr(self, attr, True)
            self.attached_cards.extend(card.attached_cards)

    def magnetize(self, targets: List['MonsterCard'], context: 'BuyPhaseContext'):
        if targets:
            targets[0].attack += self.attack
            targets[0].health += self.health
            if self.deathrattles:
                targets[0].deathrattles.extend(self.deathrattles)
            for attr in self.bool_attribute_list:
                if getattr(self, attr):
                    setattr(targets[0], attr, True)
            targets[0].attached_cards.append(self)  # TODO: BUG!!!! Replicating Menace attaches to itself
            context.owner.in_play.remove(self)

    def overkill(self, context: CombatPhaseContext):
        return

    def dissolve(self) -> List['MonsterCard']:
        if self.token:
            return [] + [type(card)() for card in self.attached_cards]
        elif self.golden:
            return [type(self)()]*3 + [type(card)() for card in self.attached_cards]
        else:
            return [type(self)()] + [type(card)() for card in self.attached_cards]

    def summon_minion_multiplier(self) -> int:
        return 1

    def deathrattle_multiplier(self) -> int:
        return 1

    def battlecry_multiplier(self) -> int:
        return 1

    def zerus_shift(self, context: 'BuyPhaseContext'):
        all_minions = PrintingPress.make_cards().unique_cards()
        random_minion = context.randomizer.select_random_minion(all_minions, context.owner.tavern.turn_count)
        if self.golden:
            random_minion.golden_transformation([])
        random_minion.shifting = True
        context.owner.hand.remove(self)
        context.owner.hand.append(random_minion)

class CardList:
    def __init__(self, cards: List[Card]):
        self.cards_by_tier = defaultdict(lambda: [])
        for card in cards:
            self.cards_by_tier[card.tier].append(card)

    def draw(self, player):
        valid_cards = []
        for tier in range(player.tavern_tier+1):
            valid_cards.extend(self.cards_by_tier[tier])
        assert valid_cards, "fnord"
        random_card = player.tavern.randomizer.select_draw_card(valid_cards, player.name, player.tavern.turn_count)
        self.cards_by_tier[random_card.tier].remove(random_card)
        return random_card

    def return_cards(self, cards: Iterator[MonsterCard]):
        for card in cards:
            self.return_card(card)

    def return_card(self, card: MonsterCard):
        self.cards_by_tier[card.tier].append(card)

    def remove_card(self, card: MonsterCard):
        self.cards_by_tier[card.tier].remove(card)

    def all_cards(self):
        return itertools.chain.from_iterable(self.cards_by_tier.values())

    def unique_cards(self):
        uniques = []
        for card in self.all_cards():
            if type(card) not in uniques:
                uniques.append(card)
        return uniques

    def __len__(self) -> int:
        return sum(len(value) for value in self.cards_by_tier.values())
