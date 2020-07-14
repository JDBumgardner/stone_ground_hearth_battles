import typing
import enum
if typing.TYPE_CHECKING:
    from hearthstone.player import Player
    from hearthstone.randomizer import Randomizer
    from hearthstone.tavern import WarParty
    from hearthstone.cards import CardEvent


class EVENTS(enum.Enum):
    SUMMON_BUY = 1
    SUMMON_COMBAT = 2
    KILL = 3
    DIES = 4
    COMBAT_START = 5
    BUY_START = 6
    ON_ATTACK = 7
    AFTER_ATTACK = 12
    SELL = 8
    BUY = 9
    BUY_END = 10
    CARD_DAMAGED = 11


class BuyPhaseContext:
    def __init__(self, owner: 'Player', randomizer: 'Randomizer'):
        self.owner = owner
        self.randomizer = randomizer


class CombatPhaseContext:
    def __init__(self, friendly_war_party: 'WarParty', enemy_war_party: 'WarParty', randomizer: 'Randomizer'):
        self.friendly_war_party = friendly_war_party
        self.enemy_war_party = enemy_war_party
        self.randomizer = randomizer

    def broadcast_combat_event(self, event: 'CardEvent'):
        #  boards are copied to prevent reindexing lists while iterating over them
        self.friendly_war_party.owner.hero.handle_event(event, self)
        for card in self.friendly_war_party.board.copy():
            # it's ok for the card to be dead
            card.handle_event(event, self)
        self.enemy_war_party.owner.hero.handle_event(event, self.enemy_context())
        for card in self.enemy_war_party.board.copy():
            card.handle_event(event, self.enemy_context())

    def enemy_context(self):
        return CombatPhaseContext(self.enemy_war_party, self.friendly_war_party, self.randomizer)
