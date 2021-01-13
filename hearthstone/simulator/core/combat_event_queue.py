import typing
from collections import deque
from typing import Optional, Tuple

from hearthstone.simulator.core.randomizer import Randomizer, DefaultRandomizer

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.cards import MonsterCard
    from hearthstone.simulator.core.combat import WarParty


class CombatEventQueue:

    def __init__(self, war_party_1: 'WarParty', war_party_2: 'WarParty', randomizer: Optional['Randomizer'] = DefaultRandomizer()):
        self.randomizer = randomizer
        self._war_party_1 = war_party_1
        self._war_party_2 = war_party_2
        self._wp1queue = deque()
        self._wp2queue = deque()
        self._queues = {self._war_party_1: self._wp1queue, self._war_party_2: self._wp2queue}

    def load_minion(self, war_party: 'WarParty', minion: 'MonsterCard'):
        self._queues[war_party].append(minion)

    def get_next_minion(self) -> Tuple['MonsterCard', 'WarParty', 'WarParty']:
        assert not self.empty()
        if self._wp1queue and self._wp2queue:
            random_queue = self.randomizer.select_event_queue(list(self._queues.values()))
            other_queue = [q for q in self._queues.values() if q != random_queue][0]
            return random_queue.popleft(), self.get_war_party(random_queue), self.get_war_party(other_queue)
        elif self._wp1queue:
            return self._wp1queue.popleft(), self._war_party_1, self._war_party_2
        else:
            return self._wp2queue.popleft(), self._war_party_2, self._war_party_1

    def empty(self) -> bool:
        return not self._wp1queue and not self._wp2queue

    def get_war_party(self, queue) -> 'WarParty':
        return list(self._queues.keys())[list(self._queues.values()).index(queue)]
