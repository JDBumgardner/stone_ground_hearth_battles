import copy
import logging
import typing
from typing import Optional, List
from hearthstone.events import CombatPhaseContext, EVENTS
from hearthstone.cards import CardEvent
if typing.TYPE_CHECKING:
    from hearthstone.player import Player
    from hearthstone.randomizer import Randomizer
    from hearthstone.cards import Card, MonsterCard


logger = logging.getLogger(__name__)


class WarParty:
    #  (HalfBoard)
    def __init__(self, player: 'Player'):
        self.owner = player
        self.board = [copy.copy(card) for card in player.in_play]
        self.next_attacker_idx = 0

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
            if not self.board[index].dead and not self.board[index].cant_attack:
                self.next_attacker_idx = index + 1
                return self.board[index]
        return None

    def get_random_monster(self, randomizer: 'Randomizer') -> Optional['MonsterCard']:
        taunt_monsters = [card for card in self.board if not card.dead and card.taunt]
        if taunt_monsters:
            return randomizer.select_attack_target(taunt_monsters)
        all_monsters = [card for card in self.board if not card.dead]
        if all_monsters:
            return randomizer.select_attack_target(all_monsters)
        return None

    def num_cards(self):
        return len(self.board)

    def summon_in_combat(self, monster: 'MonsterCard', context: CombatPhaseContext, index: Optional[int] = None):
        live_monsters_num = len([card for card in context.friendly_war_party.board if not card.dead])
        max_board_size = context.friendly_war_party.owner.maximum_board_size
        if live_monsters_num >= max_board_size:
            return
        if not index:
            index = len(context.friendly_war_party.board)
        context.friendly_war_party.board.insert(index, monster)
        if index < context.friendly_war_party.next_attacker_idx:
            context.friendly_war_party.next_attacker_idx += 1
        context.broadcast_combat_event(CardEvent(monster, EVENTS.SUMMON_COMBAT))

    def get_index(self, card):
        return self.board.index(card)

    def attackers(self) -> List['Card']:
        return [board_member for board_member in self.board if not board_member.dead and not board_member.cant_attack]


def fight_boards(war_party_1: 'WarParty', war_party_2: 'WarParty', randomizer: 'Randomizer'):
    #  Currently we are not randomizing the first to fight here
    #  Expect to pass half boards into fight_boards in random order i.e. by shuffling players in combat step
    #  Half boards are copies, the originals state cannot be changed in the combat step
    logger.debug(f"{war_party_1.owner.name}'s board is {war_party_1.board}")
    logger.debug(f"{war_party_2.owner.name}'s board is {war_party_2.board}")
    attacking_war_party = war_party_1
    defending_war_party = war_party_2
    if war_party_2.num_cards() > war_party_1.num_cards():
        attacking_war_party, defending_war_party = defending_war_party, attacking_war_party

    start_combat_event = CardEvent(None, EVENTS.COMBAT_START)
    # Friendly vs enemy warparty does not matter for broadcast_combat_event
    CombatPhaseContext(war_party_1, war_party_2, randomizer).broadcast_combat_event(start_combat_event)

    for _ in range(100):
        attacker = attacking_war_party.find_next()
        defender = defending_war_party.get_random_monster(randomizer)
        logger.debug(f'{attacking_war_party.owner.name} is attacking {defending_war_party.owner.name}')
        if not defender:
            break
        if attacker:
            start_attack(attacker, defender, attacking_war_party, defending_war_party, randomizer)
        elif not defending_war_party.attackers():
            break
        attacking_war_party, defending_war_party = defending_war_party, attacking_war_party
    damage(war_party_1, war_party_2)


def damage(half_board_1: 'WarParty', half_board_2: 'WarParty'):
    monster_damage_1 = sum([card.tier for card in half_board_1.board if not card.dead])
    monster_damage_2 = sum([card.tier for card in half_board_2.board if not card.dead])
    # Handle case where both players have cards left on board.
    if monster_damage_1 > 0 and monster_damage_2 > 0:
        logger.debug('neither player won (both players have minions left)')
    elif monster_damage_1 > 0:
        logger.debug(f'{half_board_1.owner.name} has won the fight')
        half_board_2.owner.health -= monster_damage_1 + half_board_1.owner.tavern_tier
    elif monster_damage_2 > 0:
        logger.debug(f'{half_board_2.owner.name} has won the fight')
        half_board_1.owner.health -= monster_damage_2 + half_board_2.owner.tavern_tier
    else:
        logger.debug('neither player won (no minions left)')


def start_attack(attacker: 'MonsterCard', defender: 'MonsterCard', attacking_war_party: 'WarParty', defending_war_party: 'WarParty',
                 randomizer: 'Randomizer'):
    logger.debug(f'{attacker} is attacking {defender}')
    on_attack_event = CardEvent(attacker, EVENTS.ON_ATTACK)
    combat_phase_context = CombatPhaseContext(attacking_war_party, defending_war_party, randomizer)
    combat_phase_context.broadcast_combat_event(on_attack_event)
    attacker.take_damage(defender.attack, combat_phase_context, defender, defending=False)
    defender.take_damage(attacker.attack, combat_phase_context, attacker)
    # handle "after combat" events here
    combat_phase_context.broadcast_combat_event(CardEvent(attacker, EVENTS.AFTER_ATTACK))
    attacker.resolve_death(CombatPhaseContext(attacking_war_party, defending_war_party, randomizer))
    defender.resolve_death(CombatPhaseContext(defending_war_party, attacking_war_party, randomizer))
    logger.debug(f'{attacker} has just attacked {defender}')
