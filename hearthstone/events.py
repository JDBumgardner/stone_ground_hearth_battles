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
    AFTER_ATTACK_DAMAGE = 12
    SELL = 8
    BUY = 9
    BUY_END = 10
    CARD_DAMAGED = 11
    DIVINE_SHIELD_LOST = 13
    PLAYER_DAMAGED = 14
    RETURN_TO_HAND = 15
    AFTER_ATTACK_DEATHRATTLES = 16
    END_COMBAT = 17


class BuyPhaseContext:
    def __init__(self, owner: 'Player', randomizer: 'Randomizer'):
        self.owner = owner
        self.randomizer = randomizer

    def summon_minion_multiplier(self) -> int:
        summon_multiplier = 1
        for card in self.owner.in_play:
            summon_multiplier *= card.summon_minion_multiplier()
        return summon_multiplier

    def battlecry_multiplier(self) -> int:
        battlecry_multiplier = 1
        for card in self.owner.in_play:
            battlecry_multiplier *= card.battlecry_multiplier()
        return battlecry_multiplier


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

    def summon_minion_multiplier(self) -> int:
        summon_multiplier = 1
        for card in self.friendly_war_party.board:
            if not card.dead:
                summon_multiplier *= card.summon_minion_multiplier()
        return summon_multiplier

    def deathrattle_multiplier(self) -> int:
        deathrattle_multiplier = 1
        for card in self.friendly_war_party.board:
            deathrattle_multiplier *= card.deathrattle_multiplier()
        return deathrattle_multiplier
