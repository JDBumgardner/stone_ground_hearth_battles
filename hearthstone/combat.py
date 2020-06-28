import copy
from typing import Optional

from hearthstone import events
from hearthstone.cards import Card, CardEvent, MonsterCard
from hearthstone.events import CombatPhaseContext, SUMMON_COMBAT
from hearthstone.player import Player
from hearthstone.randomizer import Randomizer


class WarParty:
    #  (HalfBoard)
    def __init__(self, player: Player):
        self.owner = player
        self.board = [copy.copy(card) for card in player.in_play]
        self.next_attacker_idx = 0

    def find_next(self) -> Optional[Card]:
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
            if not self.board[index].dead:
                self.next_attacker_idx = index + 1
                return self.board[index]
        return None

    def get_random_monster(self, randomizer: Randomizer) -> Optional[Card]:
        taunt_monsters = [card for card in self.board if not card.dead and card.taunt]
        if taunt_monsters:
            return randomizer.select_attack_target(taunt_monsters)
        all_monsters = [card for card in self.board if not card.dead]
        if all_monsters:
            return randomizer.select_attack_target(all_monsters)
        return None

    def num_cards(self):
        return len(self.board)

    def summon_in_combat(self, monster: MonsterCard, context: CombatPhaseContext, index: Optional[int] = None):
        live_monsters_num = len([card for card in context.friendly_war_party.board if not card.dead])
        max_board_size = context.friendly_war_party.owner.maximum_board_size
        if live_monsters_num >= max_board_size:
            return
        if not index:
            index = len(context.friendly_war_party.board)
        context.friendly_war_party.board.insert(index, monster)
        if index < context.friendly_war_party.next_attacker_idx:
            context.friendly_war_party.next_attacker_idx += 1
        context.broadcast_combat_event(CardEvent(monster, SUMMON_COMBAT))

    def get_index(self, card):
        return self.board.index(card)


def fight_boards(war_party_1: WarParty, war_party_2: WarParty, randomizer: Randomizer):
    #  Currently we are not randomizing the first to fight here
    #  Expect to pass half boards into fight_boards in random order i.e. by shuffling players in combat step
    #  Half boards are copies, the originals state cannot be changed in the combat step
    attacking_war_party = war_party_1
    defending_war_party = war_party_2
    if war_party_2.num_cards() > war_party_1.num_cards():
        attacking_war_party, defending_war_party = defending_war_party, attacking_war_party

    start_combat_event = CardEvent(None, events.COMBAT_START)
    # Friendly vs enemy warparty does not matter for broadcast_combat_event
    CombatPhaseContext(war_party_1, war_party_2, randomizer).broadcast_combat_event(start_combat_event)

    while True:
        attacker = attacking_war_party.find_next()
        defender = defending_war_party.get_random_monster(randomizer)
        if not attacker or not defender:
            break
        print(f'{attacking_war_party.owner.name} is attacking {defending_war_party.owner.name}')
        start_attack(attacker, defender, attacking_war_party, defending_war_party, randomizer)
        attacking_war_party, defending_war_party = defending_war_party, attacking_war_party
    damage(war_party_1, war_party_2)


def damage(half_board_1: WarParty, half_board_2: WarParty):
    monster_damage_1 = sum([card.tier for card in half_board_1.board if not card.dead])
    monster_damage_2 = sum([card.tier for card in half_board_2.board if not card.dead])
    if monster_damage_1 > 0:
        print(f'{half_board_1.owner.name} has won the fight')
        half_board_2.owner.health -= monster_damage_1 + half_board_1.owner.tavern_tier
    elif monster_damage_2 > 0:
        print(f'{half_board_2.owner.name} has won the fight')
        half_board_1.owner.health -= monster_damage_2 + half_board_2.owner.tavern_tier
    else:
        print('neither player won')


def start_attack(attacker: Card, defender: Card, attacking_war_party: WarParty, defending_war_party: WarParty,
                 randomizer: Randomizer):
    print(f'{attacker} is attacking {defender}')
    on_attack_event = CardEvent(attacker, events.ON_ATTACK)
    CombatPhaseContext(attacking_war_party, defending_war_party, randomizer).broadcast_combat_event(on_attack_event)
    attacker.take_damage(defender.attack)
    defender.take_damage(attacker.attack)

    # handle "after combat" events here

    attacker.resolve_death(CombatPhaseContext(attacking_war_party, defending_war_party, randomizer))
    defender.resolve_death(CombatPhaseContext(defending_war_party, attacking_war_party, randomizer))
    print(f'{attacker} has just attacked {defender}')
