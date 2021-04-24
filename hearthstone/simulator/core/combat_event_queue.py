import typing
from collections import deque
from typing import Optional, Tuple

import autoslot

from hearthstone.simulator.core.events import EVENTS
from hearthstone.simulator.core.randomizer import Randomizer, DefaultRandomizer

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.cards import MonsterCard
    from hearthstone.simulator.core.combat import WarParty


class CombatEventQueue(autoslot.Slots):
    def __init__(self, war_party_1: 'WarParty', war_party_2: 'WarParty',
                 randomizer: Optional['Randomizer'] = None):
        self.randomizer = randomizer or DefaultRandomizer()
        self.queues = {
            EVENTS.DEATHRATTLE_TRIGGERED: {war_party_1: deque(), war_party_2: deque()},
            EVENTS.DIES: {war_party_1: deque(), war_party_2: deque()}
        }

    def load_minion(self, event: 'EVENTS', war_party: 'WarParty', minion: 'MonsterCard',
                    foe: Optional['MonsterCard'] = None):
        self.queues[event][war_party].append((minion, foe))

    def get_next_minion(self, event: 'EVENTS') -> Tuple['MonsterCard', Optional['MonsterCard'], 'WarParty', 'WarParty']:
        assert not self.all_empty()

        if all(bool(queue) for queue in self.queues[event].values()):
            non_empty_queue = self.randomizer.select_event_queue(list(self.queues[event].values()))
        else:
            non_empty_queue = [q for q in self.queues[event].values() if bool(q)][0]

        other_queue = [q for q in self.queues[event].values() if q != non_empty_queue][0]
        friendly_war_party = self.get_war_party(non_empty_queue, event)
        enemy_war_party = self.get_war_party(other_queue, event)
        minion, foe = non_empty_queue.popleft()

        return minion, foe, friendly_war_party, enemy_war_party

    def all_empty(self) -> bool:
        return all(not bool(queue) for pairs in self.queues.values() for queue in pairs.values())

    def event_empty(self, event: 'EVENTS') -> bool:
        return all(not bool(queue) for queue in self.queues[event].values())

    def get_war_party(self, queue: deque, event: 'EVENTS') -> 'WarParty':
        queues_of_event = list(self.queues[event].keys())
        queue_index = list(self.queues[event].values()).index(queue)
        return queues_of_event[queue_index]
