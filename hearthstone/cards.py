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

        return "{" + rep + "}"

    def take_damage(self, damage: int, combat_phase_context: CombatPhaseContext):
        if self.divine_shield and not damage <= 0:
            self.divine_shield = False
            combat_phase_context.broadcast_combat_event(CardEvent(self, EVENTS.DIVINE_SHIELD_LOST))
        else:
            self.health -= damage
            combat_phase_context.broadcast_combat_event(CardEvent(self, EVENTS.CARD_DAMAGED))

    def resolve_death(self, context: CombatPhaseContext):
        if self.health <= 0:
            self.dead = True
            card_death_event = CardEvent(self, EVENTS.DIES)
            context.broadcast_combat_event(card_death_event)
            if self.reborn:
                self.resolve_reborn()

    def resolve_reborn(self):
        self.dead = False
        self.attack = self.base_attack * 2 if self.golden else self.base_attack
        self.health = 1
        self.reborn = False
        self.divine_shield = self.base_divine_shield
        self.magnetic = self.base_magnetic
        self.poisonous = self.base_poisonous
        self.taunt = self.base_taunt
        self.windfury = self.base_windfury
        self.cleave = self.base_cleave

    def change_state(self, new_state):
        self.tavern.run_callbacks(self, new_state)
        self.state = new_state

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        if self == event.card:
            if event.event is EVENTS.DIES:
                for deathrattle in self.deathrattles:
                    deathrattle(self, context)
            elif event.event is EVENTS.SUMMON_BUY:
                if self.battlecry:
                    self.battlecry(event.targets, context)
                if event.card.tracked:
                    context.owner.counted_cards[type(event.card)] += 1
        if not self.dead:
            self.handle_event_powers(event, context)

    def handle_event_in_hand(self, event: CardEvent, context: BuyPhaseContext):
        return

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

    def magnetic_transformation(self, magnetic_card: 'MonsterCard'):
        self.attack += magnetic_card.base_attack
        self.health += magnetic_card.base_health
        if magnetic_card.base_deathrattle:
            self.deathrattles.extend(magnetic_card.deathrattles[1:])
        else:
            self.deathrattles.extend(magnetic_card.deathrattles)
        for attr in magnetic_card.bool_attribute_list:
            if getattr(magnetic_card, attr):
                setattr(self, attr, True)
        self.attached_cards.append(type(magnetic_card)())

    def overkill(self):
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
