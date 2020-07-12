from typing import Set, List, Optional, Callable, Type, Union
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
    def __init__(self, card: Optional['MonsterCard'], event: int, targets: Optional[List['MonsterCard']] = None):
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
    monster_type = ""
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
        else:
            self.health -= damage
            combat_phase_context.broadcast_combat_event(CardEvent(self, events.CARD_DAMAGED))

    def resolve_death(self, context: CombatPhaseContext):
        if self.health <= 0:
            self.dead = True
            card_death_event = CardEvent(self, EVENTS.DIES.value)
            context.broadcast_combat_event(card_death_event)

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

    def overkill(self):
        return

    def dissolve(self) -> List['MonsterCard']:
        if self.token:
            return []
        elif self.golden:
            return [type(self)()]*3
        else:
            return [type(self)()]


class CardList:
    def __init__(self, cards: List[Card]):
        self.cards: List[Card] = list(cards)

    def draw(self, player):
        valid_cards = [card for card in self.cards if player.tavern_tier >= card.tier]
        assert valid_cards, "fnord"
        random_card = player.tavern.randomizer.select_draw_card(valid_cards, player.name, player.tavern.turn_count)
        self.cards.remove(random_card)
        return random_card

    def __len__(self) -> int:
        return len(self.cards)
