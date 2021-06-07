import unittest

from hearthstone.simulator.core.card_pool import *
from hearthstone.simulator.core.combat import WarParty
from hearthstone.simulator.core.combat_event_queue import CombatEventQueue
from hearthstone.simulator.core.events import EVENTS
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.testing.battlegrounds_test_case import BattleGroundsTestCase


class CombatEventQueueTest(BattleGroundsTestCase):
    def test_empty(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Player 1")
        player_2 = tavern.add_player_with_hero("Player 2")
        war_party_1 = WarParty(player_1)
        war_party_2 = WarParty(player_2)
        q = CombatEventQueue(war_party_1, war_party_2)
        self.assertTrue(q.all_empty())
        self.assertTrue(q.event_empty(EVENTS.DEATHRATTLE_TRIGGERED))
        self.assertTrue(q.event_empty(EVENTS.DIES))

    def test_load_deathrattle(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Player 1")
        player_2 = tavern.add_player_with_hero("Player 2")
        war_party_1 = WarParty(player_1)
        war_party_2 = WarParty(player_2)
        q = CombatEventQueue(war_party_1, war_party_2)
        q.load_minion(EVENTS.DEATHRATTLE_TRIGGERED, war_party_1, KaboomBot())
        self.assertIn(KaboomBot, [type(pair[0]) for pair in q.queues[EVENTS.DEATHRATTLE_TRIGGERED][war_party_1]])

    def test_load_dies(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Player 1")
        player_2 = tavern.add_player_with_hero("Player 2")
        war_party_1 = WarParty(player_1)
        war_party_2 = WarParty(player_2)
        q = CombatEventQueue(war_party_1, war_party_2)
        q.load_minion(EVENTS.DIES, war_party_1, AlleyCat())
        self.assertIn(AlleyCat, [type(pair[0]) for pair in q.queues[EVENTS.DIES][war_party_1]])

    def test_deathrattle_queue_not_empty(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Player 1")
        player_2 = tavern.add_player_with_hero("Player 2")
        war_party_1 = WarParty(player_1)
        war_party_2 = WarParty(player_2)
        q = CombatEventQueue(war_party_1, war_party_2)
        q.load_minion(EVENTS.DEATHRATTLE_TRIGGERED, war_party_1, KaboomBot())
        self.assertFalse(q.all_empty())
        self.assertFalse(q.event_empty(EVENTS.DEATHRATTLE_TRIGGERED))
        self.assertTrue(q.event_empty(EVENTS.DIES))

    def test_dies_queue_not_empty(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Player 1")
        player_2 = tavern.add_player_with_hero("Player 2")
        war_party_1 = WarParty(player_1)
        war_party_2 = WarParty(player_2)
        q = CombatEventQueue(war_party_1, war_party_2)
        q.load_minion(EVENTS.DIES, war_party_1, AlleyCat())
        self.assertFalse(q.all_empty())
        self.assertFalse(q.event_empty(EVENTS.DIES))
        self.assertTrue(q.event_empty(EVENTS.DEATHRATTLE_TRIGGERED))

    def test_get_next_deathrattle(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Player 1")
        player_2 = tavern.add_player_with_hero("Player 2")
        war_party_1 = WarParty(player_1)
        war_party_2 = WarParty(player_2)
        q = CombatEventQueue(war_party_1, war_party_2)
        q.load_minion(EVENTS.DEATHRATTLE_TRIGGERED, war_party_1, KaboomBot())
        tup = q.get_next_minion(EVENTS.DEATHRATTLE_TRIGGERED)
        self.assertIsInstance(tup[0], KaboomBot)
        self.assertTupleEqual((None, war_party_1, war_party_2), tup[1:])

    def test_get_next_dies(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Player 1")
        player_2 = tavern.add_player_with_hero("Player 2")
        war_party_1 = WarParty(player_1)
        war_party_2 = WarParty(player_2)
        q = CombatEventQueue(war_party_1, war_party_2)
        q.load_minion(EVENTS.DIES, war_party_1, AlleyCat())
        tup = q.get_next_minion(EVENTS.DIES)
        self.assertIsInstance(tup[0], AlleyCat)
        self.assertTupleEqual((None, war_party_1, war_party_2), tup[1:])


if __name__ == '__main__':
    unittest.main()
