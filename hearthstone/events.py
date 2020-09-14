import typing
import enum

from typing import Optional, List

if typing.TYPE_CHECKING:
    from hearthstone.player import Player
    from hearthstone.randomizer import Randomizer
    from hearthstone.tavern import WarParty
    from hearthstone.cards import MonsterCard


class EVENTS(enum.Enum):
    SUMMON_BUY = 1
    SUMMON_COMBAT = 2
    KILL = 3
    DIES = 4
    COMBAT_START = 5
    BUY_START = 6
    ON_ATTACK = 7
    SELL = 8
    BUY = 9
    BUY_END = 10
    CARD_DAMAGED = 11
    AFTER_ATTACK_DAMAGE = 12
    DIVINE_SHIELD_LOST = 13
    PLAYER_DAMAGED = 14
    AFTER_ATTACK_DEATHRATTLES = 15
    END_COMBAT = 16
    TAVERN_UPGRADE = 17


class CardEvent:
    def __init__(self, eventid: EVENTS):
        self.event = eventid
        self.card = None


class SummonBuyEvent(CardEvent):
    def __init__(self, card: 'MonsterCard', targets: Optional[List['MonsterCard']] = None):
        super().__init__(EVENTS.SUMMON_BUY)
        self.card = card
        self.targets = targets


class SummonCombatEvent(CardEvent):
    def __init__(self, card: 'MonsterCard'):
        super().__init__(EVENTS.SUMMON_COMBAT)
        self.card = card


class DiesEvent(CardEvent):
    def __init__(self, card: 'MonsterCard', foe: Optional['MonsterCard']):
        super().__init__(EVENTS.DIES)
        self.card = card
        self.foe = foe


class CombatStartEvent(CardEvent):
    def __init__(self):
        super().__init__(EVENTS.COMBAT_START)


class BuyStartEvent(CardEvent):
    def __init__(self):
        super().__init__(EVENTS.BUY_START)


class OnAttackEvent(CardEvent):
    def __init__(self, card: 'MonsterCard', foe: 'MonsterCard'):
        super().__init__(EVENTS.ON_ATTACK)
        self.card = card
        self.foe = foe


class AfterAttackDamageEvent(CardEvent):
    def __init__(self, card: 'MonsterCard', foe: 'MonsterCard'):
        super().__init__(EVENTS.AFTER_ATTACK_DAMAGE)
        self.card = card
        self.foe = foe


class SellEvent(CardEvent):
    def __init__(self, card: 'MonsterCard'):
        super().__init__(EVENTS.SELL)
        self.card = card


class BuyEvent(CardEvent):
    def __init__(self, card: 'MonsterCard'):
        super().__init__(EVENTS.BUY)
        self.card = card


class BuyEndEvent(CardEvent):
    def __init__(self):
        super().__init__(EVENTS.BUY_END)


class CardDamagedEvent(CardEvent):
    def __init__(self, card: 'MonsterCard', foe: 'MonsterCard'):
        super().__init__(EVENTS.CARD_DAMAGED)
        self.card = card
        self.foe = foe


class DivineShieldLostEvent(CardEvent):
    def __init__(self, card: 'MonsterCard', foe: 'MonsterCard'):
        super().__init__(EVENTS.DIVINE_SHIELD_LOST)
        self.card = card
        self.foe = foe


class PlayerDamagedEvent(CardEvent):
    def __init__(self):
        super().__init__(EVENTS.PLAYER_DAMAGED)


class AfterAttackDeathrattleEvent(CardEvent):
    def __init__(self, card: 'MonsterCard', foe: 'MonsterCard'):
        super().__init__(EVENTS.AFTER_ATTACK_DEATHRATTLES)
        self.card = card
        self.foe = foe


class EndCombatEvent(CardEvent):
    def __init__(self, won_combat: bool):
        super().__init__(EVENTS.END_COMBAT)
        self.won_combat = won_combat


class TavernUpgradeEvent(CardEvent):
    def __init__(self):
        super().__init__(EVENTS.TAVERN_UPGRADE)


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


