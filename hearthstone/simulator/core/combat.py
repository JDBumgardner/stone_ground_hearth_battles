import copy
import logging
import typing
from typing import Optional, List, Set

from hearthstone.simulator.core import events
from hearthstone.simulator.core.combat_event_queue import CombatEventQueue
from hearthstone.simulator.core.events import CombatPhaseContext, EVENTS

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.player import Player
    from hearthstone.simulator.core.randomizer import Randomizer
    from hearthstone.simulator.core.cards import MonsterCard


logger = logging.getLogger(__name__)


class WarParty:
    #  (HalfBoard)
    def __init__(self, player: 'Player'):
        self.owner = player
        self.board = []
        for card in player.in_play:  # TODO: better way to link cards in combat with cards in the buy phase?
            card_copy = copy.copy(card)
            card_copy.link = card
            self.board.append(card_copy)
        self.next_attacker_idx = 0
        self.dead_minions: List[MonsterCard] = []

    def __repr__(self):
        return f"{self.owner}'s WarParty"

    def find_next(self) -> Optional['MonsterCard']:
        #  Sets the index for the next monster who will fight from your side.
        #  Must be called after active player monster fights
        #  Also after a monster dies from combat if it was the active monster
        #  Take care not to call this function twice after a monster dies fighting
        #  The final condition indicates a next fighting monster cannot be found
        #  Meaning the player has lost the round
        #  Take care to handle the case where both players die in the same monster fight action
        num_cards = len(self.board)
        for offset in range(0, num_cards):
            index = (self.next_attacker_idx + offset) % num_cards
            if not self.board[index].dead and not self.board[index].cant_attack and not self.board[index].attack <= 0:
                self.next_attacker_idx = index + 1
                return self.board[index]
        return None

    def get_attack_target(self, randomizer: 'Randomizer', attacker: Optional['MonsterCard'] = None) -> Optional['MonsterCard']:
        if attacker is None:
            return None
        attack_targets = attacker.valid_attack_targets(self.live_minions())
        if attack_targets:
            return randomizer.select_attack_target(attack_targets)
        return None

    def live_minions(self) -> List['MonsterCard']:
        return [card for card in self.board if not card.dead]

    def num_cards(self):
        return len(self.board)

    def summon_in_combat(self, monster: 'MonsterCard', context: 'CombatPhaseContext', index: Optional[int] = None):
        if not context.friendly_war_party.room_on_board():
            return
        if not index:
            index = len(context.friendly_war_party.board)
        context.friendly_war_party.board.insert(index, monster)
        if index < context.friendly_war_party.next_attacker_idx:
            context.friendly_war_party.next_attacker_idx += 1
        context.broadcast_combat_event(events.SummonCombatEvent(monster))

    def get_index(self, card):
        return self.board.index(card)

    def attackers(self) -> List['MonsterCard']:
        return [board_member for board_member in self.board if not board_member.dead and not board_member.cant_attack and board_member.attack > 0]

    def room_on_board(self) -> bool:
        return len(self.live_minions()) < self.owner.maximum_board_size

    def adjacent_minions(self, minion: 'MonsterCard') -> List['MonsterCard']:
        if minion.dead:
            living_with_minion = [card for card in self.board if not card.dead or card == minion]
            return [card for card in living_with_minion if
                    abs(living_with_minion.index(card) - living_with_minion.index(minion)) == 1]
        else:
            return [card for card in self.live_minions() if
                    abs(self.live_minions().index(card) - self.live_minions().index(minion)) == 1]

    def get_defenders(self, attacker: 'MonsterCard', defender: 'MonsterCard') -> List['MonsterCard']:
        if attacker.cleave:
            defenders = self.adjacent_minions(defender)
            defenders.insert(1, defender)
        else:
            defenders = [defender]
        return defenders

    def has_dying_minions(self) -> bool:
        return bool([minion for minion in self.board if minion.is_dying()])


def fight_boards(war_party_1: 'WarParty', war_party_2: 'WarParty', randomizer: 'Randomizer'):
    #  Currently we are not randomizing the first to fight here
    #  Expect to pass half boards into fight_boards in random order i.e. by shuffling players in combat step
    #  Half boards are copies, the originals state cannot be changed in the combat step
    combat_event_queue = CombatEventQueue(war_party_1, war_party_2, randomizer)
    damaged_minions = set()
    context = CombatPhaseContext(war_party_1, war_party_2, randomizer, combat_event_queue, damaged_minions)
    context.broadcast_combat_event(events.CombatPrePhaseEvent())
    logger.debug(
        f"{war_party_1.owner} (tier {war_party_1.owner.tavern_tier}, {war_party_1.owner.health} health) is fighting {war_party_2.owner} (tier {war_party_2.owner.tavern_tier}, {war_party_2.owner.health} health)")
    logger.debug(f"{war_party_1.owner}'s board is {war_party_1.board}")
    logger.debug(f"{war_party_2.owner}'s board is {war_party_2.board}")
    attacking_war_party = war_party_1
    defending_war_party = war_party_2
    if war_party_2.num_cards() > war_party_1.num_cards():
        attacking_war_party, defending_war_party = defending_war_party, attacking_war_party

    # Friendly vs enemy warparty does not matter for broadcast_combat_event
    context.broadcast_combat_event(events.CombatStartEvent())

    for _ in range(100):
        attacker = attacking_war_party.find_next()
        num_attacks = attacker.num_attacks() if attacker else 1
        for _ in range(num_attacks):
            defender = defending_war_party.get_attack_target(randomizer, attacker)
            if defender is None:
                break
            if attacker and not attacker.dead:
                logger.debug(f'{attacking_war_party.owner} is attacking {defending_war_party.owner}')
                start_attack(attacker, defender, attacking_war_party, defending_war_party, randomizer, combat_event_queue, damaged_minions)
        if not defending_war_party.attackers():
            break
        attacking_war_party, defending_war_party = defending_war_party, attacking_war_party
    player_damage(war_party_1, war_party_2, randomizer)
    war_party_1.owner.last_opponent_warband = [card.copy() for card in war_party_2.owner.in_play]
    war_party_2.owner.last_opponent_warband = [card.copy() for card in war_party_1.owner.in_play]


def player_damage(half_board_1: 'WarParty', half_board_2: 'WarParty', randomizer: 'Randomizer'):
    monster_damage_1 = sum([card.tier for card in half_board_1.board if not card.dead])
    monster_damage_2 = sum([card.tier for card in half_board_2.board if not card.dead])
    # Handle case where both players have cards left on board.
    if monster_damage_1 > 0 and monster_damage_2 > 0:
        logger.debug('neither player won (both players have minions left)')
        half_board_1.owner.broadcast_buy_phase_event(events.EndCombatEvent(won_combat=False), randomizer)
        half_board_2.owner.broadcast_buy_phase_event(events.EndCombatEvent(won_combat=False), randomizer)
        half_board_1.owner.broadcast_global_event(events.ResultsBroadcastEvent(tie=True))
    elif monster_damage_1 > 0:
        logger.debug(f'{half_board_1.owner} has won the fight')
        logger.debug(f'{half_board_2.owner} took {monster_damage_1 + half_board_1.owner.tavern_tier} damage.')
        logger.debug(f"{half_board_1.owner}'s remaining board: {[card for card in half_board_1.board if not card.dead]}")
        damage_dealt = monster_damage_1 + half_board_1.owner.tavern_tier
        half_board_2.owner.health -= damage_dealt
        half_board_1.owner.broadcast_buy_phase_event(events.EndCombatEvent(won_combat=True), randomizer)
        half_board_2.owner.broadcast_buy_phase_event(events.EndCombatEvent(won_combat=False, damage_taken=damage_dealt), randomizer)
        half_board_1.owner.broadcast_global_event(events.ResultsBroadcastEvent(winner=half_board_1.owner, loser=half_board_2.owner))
    elif monster_damage_2 > 0:
        logger.debug(f'{half_board_2.owner} has won the fight')
        logger.debug(f'{half_board_1.owner} took {monster_damage_2 + half_board_2.owner.tavern_tier} damage.')
        logger.debug(f"{half_board_2.owner}'s remaining board: {[card for card in half_board_2.board if not card.dead]}")
        damage_dealt = monster_damage_2 + half_board_2.owner.tavern_tier
        half_board_1.owner.health -= damage_dealt
        half_board_1.owner.broadcast_buy_phase_event(events.EndCombatEvent(won_combat=False, damage_taken=damage_dealt), randomizer)
        half_board_2.owner.broadcast_buy_phase_event(events.EndCombatEvent(won_combat=True), randomizer)
        half_board_1.owner.broadcast_global_event(events.ResultsBroadcastEvent(winner=half_board_2.owner, loser=half_board_1.owner))
    else:
        logger.debug('neither player won (no minions left)')
        half_board_1.owner.broadcast_buy_phase_event(events.EndCombatEvent(won_combat=False), randomizer)
        half_board_2.owner.broadcast_buy_phase_event(events.EndCombatEvent(won_combat=False), randomizer)
        half_board_1.owner.broadcast_global_event(events.ResultsBroadcastEvent(tie=True))


def start_attack(attacker: 'MonsterCard', defender: 'MonsterCard', attacking_war_party: 'WarParty', defending_war_party: 'WarParty',
                 randomizer: 'Randomizer', event_queue: 'CombatEventQueue', damaged_minions: Set['MonsterCard']):
    logger.debug(f'{attacker} is attacking {defender}')
    combat_phase_context = CombatPhaseContext(attacking_war_party, defending_war_party, randomizer, event_queue, damaged_minions)
    combat_phase_context.enemy_context().broadcast_combat_event(events.IsAttackedEvent(defender))
    combat_phase_context.broadcast_combat_event(events.OnAttackEvent(attacker))

    taunt_diversions = [card for card in defending_war_party.live_minions() if card.divert_taunt_attack]
    if defender.taunt and taunt_diversions:
        new_defender = randomizer.select_attack_target(taunt_diversions)
        logger.debug(f'{new_defender} has diverted an attack on {defender}')
        combat_phase_context.enemy_context().broadcast_combat_event(events.IsAttackedEvent(new_defender))
        defender = new_defender

    attacker.take_damage(defender.attack, combat_phase_context, defender, defending=False)
    for enemy in defending_war_party.get_defenders(attacker, defender):
        enemy.take_damage(attacker.attack, combat_phase_context.enemy_context(), attacker)
    # handle "after combat" events here
    combat_phase_context.broadcast_combat_event(events.AfterAttackDamageEvent(attacker, foe=defender))
    resolve_combat_events(randomizer, event_queue, attacking_war_party, defending_war_party, damaged_minions)
    combat_phase_context.broadcast_combat_event(events.AfterAttackDeathrattleEvent(defender, foe=attacker))
    logger.debug(f'{attacker} has just attacked {defender}')


def resolve_combat_events(randomizer: 'Randomizer', event_queue: 'CombatEventQueue', attacking_war_party: 'WarParty',
                          defending_war_party: 'WarParty', damaged_minions: Set['MonsterCard']):
    while attacking_war_party.has_dying_minions() or defending_war_party.has_dying_minions():

        mark_combat_deaths(attacking_war_party, defending_war_party, event_queue, damaged_minions)

        while not event_queue.event_empty(EVENTS.DIES):
            minion, foe, friendly_war_party, enemy_war_party = event_queue.get_next_minion(EVENTS.DIES)
            context = CombatPhaseContext(friendly_war_party, enemy_war_party, randomizer, event_queue, damaged_minions)
            context.broadcast_combat_event(events.DiesEvent(minion, foe))

        while not event_queue.event_empty(EVENTS.DEATHRATTLE_TRIGGERED):
            minion, _, friendly_war_party, enemy_war_party = event_queue.get_next_minion(EVENTS.DEATHRATTLE_TRIGGERED)
            context = CombatPhaseContext(friendly_war_party, enemy_war_party, randomizer, event_queue, damaged_minions)
            for _ in range(context.deathrattle_multiplier()):
                for deathrattle in minion.deathrattles:
                    deathrattle(minion, context)


def mark_combat_deaths(attacking_war_party: 'WarParty', defending_war_party: 'WarParty',
                       event_queue: 'CombatEventQueue', damaged_minions: Set['MonsterCard']):
    marked_dead_minions = set()
    for minion in damaged_minions:
        if minion.is_dying():
            minion.dead = True
            war_party = attacking_war_party if minion in attacking_war_party.board else defending_war_party
            war_party.dead_minions.append(minion)
            event_queue.load_minion(EVENTS.DIES, war_party, minion, minion.dealt_lethal_damage_by)
            marked_dead_minions.add(minion)
    damaged_minions -= marked_dead_minions
