import unittest
from typing import List, Tuple, Type

from hearthstone.card_pool import *
from hearthstone.cards import Card, CardType, PrintingPress
from hearthstone.hero_pool import *
from hearthstone.player import StoreIndex, HandIndex, BoardIndex
from hearthstone.randomizer import DefaultRandomizer
from hearthstone.tavern import Tavern


def force_card(cards: List[Card], card_type) -> Card:
    return [card for card in cards if isinstance(card, card_type)][0]


class CardForcer(DefaultRandomizer):
    def __init__(self, forced_cards: List[CardType]):
        self.forced_cards = forced_cards

    def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
        next_card_type = self.forced_cards.pop(0)
        return force_card(cards, next_card_type)


class RepeatedCardForcer(DefaultRandomizer):
    def __init__(self, repeatedly_forced_cards: List[CardType]):
        self.repeatedly_forced_cards = repeatedly_forced_cards
        self.pointer = 0

    def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
        next_card_type = self.repeatedly_forced_cards[self.pointer]
        self.pointer = (self.pointer + 1) % len(self.repeatedly_forced_cards)
        return force_card(cards, next_card_type)


class CardTests(unittest.TestCase):
    def assertCardListEquals(self, cards, expected, msg=None):
        self.assertListEqual([type(card) for card in cards], expected, msg=msg)

    def test_default_cardlist(self):
        default_cardlist = PrintingPress.make_cards()
        self.assertGreater(len(default_cardlist), 20)
        print(f"the length of the default cardlist is {len(default_cardlist)}.")

    def test_draw(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(len(player_1.store), 3)
        self.assertEqual(len(player_2.store), 3)
        self.assertNotIn(None, player_1.store)
        self.assertNotIn(None, player_2.store)

    def type_of_cards(self, cards: List[Card]) -> List[Type]:
        return list(map(type, cards))

    class TestGameRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            if player_name == "Dante_Kong":
                return force_card(cards, WrathWeaver)
            if player_name == "lucy":
                return force_card(cards, RighteousProtector)

    def upgrade_to_tier(self, tavern: Tavern, tier: int):
        players = list(tavern.players.values())
        while players[0].tavern_tier < tier:
            tavern.buying_step()
            if players[0].coins >= players[0].tavern_upgrade_cost:
                for player in players:
                    player.upgrade_tavern()
            tavern.combat_step()

    def test_game(self):
        tavern = Tavern()
        tavern.randomizer = self.TestGameRandomizer()
        deck_length_pre = len(tavern.deck)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(len(tavern.deck), deck_length_pre - 6)
        self.assertCardListEquals(player_1.store, [WrathWeaver, WrathWeaver, WrathWeaver])
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_2.store, [RighteousProtector, RighteousProtector, RighteousProtector])
        self.assertCardListEquals(player_2.hand, [])
        self.assertCardListEquals(player_2.in_play, [])
        for player_name, player in tavern.players.items():
            self.assertEqual(player.coins, 3, f"{player.name} does not have the right number of coins")
        player_1.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.store, [WrathWeaver, WrathWeaver])
        self.assertCardListEquals(player_1.hand, [WrathWeaver])
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_2.store, [RighteousProtector, RighteousProtector])
        self.assertCardListEquals(player_2.hand, [RighteousProtector])
        self.assertCardListEquals(player_2.in_play, [])
        for player_name, player in tavern.players.items():
            self.assertEqual(player.coins, 0, f"{player.name} does not have the right number of coins")
        player_1.summon_from_hand(HandIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.store, [WrathWeaver, WrathWeaver])
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [WrathWeaver])
        self.assertCardListEquals(player_2.store, [RighteousProtector, RighteousProtector])
        self.assertCardListEquals(player_2.hand, [])
        self.assertCardListEquals(player_2.in_play, [RighteousProtector])
        tavern.combat_step()
        self.assertEqual(player_1.health, 38, f"{player_1.name}'s heath is incorrect")
        self.assertEqual(player_2.health, 40, f"{player_2.name}'s heath is incorrect")

    class TestTwoRoundsRandomizer(DefaultRandomizer):
        def __init__(self):
            self.cards_drawn = 0

        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            if round_number == 0:
                if player_name == "Dante_Kong":
                    return force_card(cards, WrathWeaver)

                if player_name == "lucy":
                    return force_card(cards, RighteousProtector)
            elif round_number == 1:
                if player_name == "Dante_Kong":
                    return force_card(cards, WrathWeaver)

                if player_name == "lucy":
                    return force_card(cards, WrathWeaver)

        def select_attack_target(self, defenders: List[Card]) -> Card:
            target = [card for card in defenders if type(card) is not RighteousProtector]
            if target:
                return target[0]
            else:
                return defenders[0]

        def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
            number_of_battles = len(players) // 2
            return list(zip(players[:number_of_battles], players[number_of_battles:]))

    def test_two_rounds(self):
        tavern = Tavern()
        tavern.randomizer = self.TestTwoRoundsRandomizer()
        deck_length_pre = len(tavern.deck)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(len(tavern.deck), deck_length_pre - 6)
        self.assertCardListEquals(player_1.store, [WrathWeaver, WrathWeaver, WrathWeaver])
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_2.store, [RighteousProtector, RighteousProtector, RighteousProtector])
        self.assertCardListEquals(player_2.hand, [])
        self.assertCardListEquals(player_2.in_play, [])
        for player_name, player in tavern.players.items():
            self.assertEqual(player.coins, 3, f"{player.name} does not have the right number of coins")
        player_1.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.store, [WrathWeaver, WrathWeaver])
        self.assertCardListEquals(player_1.hand, [WrathWeaver])
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_2.store, [RighteousProtector, RighteousProtector])
        self.assertCardListEquals(player_2.hand, [RighteousProtector])
        self.assertCardListEquals(player_2.in_play, [])
        for player_name, player in tavern.players.items():
            self.assertEqual(player.coins, 0, f"{player.name} does not have the right number of coins")
        player_1.summon_from_hand(HandIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.store, [WrathWeaver, WrathWeaver])
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [WrathWeaver])
        self.assertCardListEquals(player_2.store, [RighteousProtector, RighteousProtector])
        self.assertCardListEquals(player_2.hand, [])
        self.assertCardListEquals(player_2.in_play, [RighteousProtector])
        tavern.combat_step()
        self.assertEqual(player_1.health, 38, f"{player_1.name}'s heath is incorrect")
        self.assertEqual(player_2.health, 40, f"{player_2.name}'s heath is incorrect")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertEqual(player_1.health, 36, f"{player_1.name}'s heath is incorrect")
        self.assertEqual(player_2.health, 40, f"{player_2.name}'s heath is incorrect")

    class TestBattlecryRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            return [card for card in cards if type(card) is AlleyCat][0]

    def test_battlecry(self):
        tavern = Tavern()
        tavern.randomizer = self.TestBattlecryRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertListEqual([AlleyCat, TabbyCat], self.type_of_cards(player_1.in_play))

    class TestRaftWeaverRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            if round_number == 0:
                return [card for card in cards if type(card) is WrathWeaver][0]
            else:
                return [card for card in cards if type(card) is FiendishServant][0]

    def test_raft_weaver(self):  # TODO: rename
        tavern = Tavern()
        tavern.randomizer = self.TestRaftWeaverRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        print("the first monster in your hand is ", player_1.hand[0])
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        print("the first monster in your hand is ", player_1.hand[0])
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(39, player_1.health)
        self.assertEqual(player_1.in_play[0].attack, 3)


    def test_sell_from_board(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([MicroMachine] * 6)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.coins, 0)
        player_1.sell_board_minion(BoardIndex(0))
        self.assertEqual(player_1.coins, 1)

    def test_pyramad(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong", Pyramad())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.summon_from_hand(HandIndex(0))
        prior_health = [card.health for card in player_1.in_play]
        player_1.hero_power()
        print(player_1.in_play[0])
        pyramad_bonus = 0
        for i, card in enumerate(player_1.in_play):
            if card.health == prior_health[i] + 4:
                pyramad_bonus += 1
            else:
                self.assertEqual(card.health, prior_health[i])
        self.assertEqual(pyramad_bonus, 1)

    class TestLordJaraxxusRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            return [card for card in cards if type(card) is WrathWeaver][0]

    def test_lord_jaraxxus(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([FiendishServant, RighteousProtector, RighteousProtector] * 4)
        player_1 = tavern.add_player_with_hero("Dante_Kong", LordJaraxxus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()

        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))

        self.assertCardListEquals(player_1.in_play, [FiendishServant, RighteousProtector])
        player_1.hero_power()
        fiendish_servant_stats = (player_1.in_play[0].attack, player_1.in_play[0].health)
        righteous_protector_stats = (player_1.in_play[1].attack, player_1.in_play[1].health)
        self.assertEqual(fiendish_servant_stats, (3, 2))
        self.assertEqual(righteous_protector_stats, (1, 1))

    def test_patchwerk(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong", PatchWerk())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(player_1.health, 50)
        self.assertEqual(player_2.health, 40)

    def test_nefarian(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([AlleyCat] * 6)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Nefarian())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power()
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 40)

    def test_card_triple(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([AlleyCat] * 18)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [AlleyCat])
        self.assertEqual(player_1.hand[0].golden, True)
        self.assertCardListEquals(player_1.in_play, [TabbyCat])
        self.assertEqual(player_1.in_play[0].golden, False)
        self.assertEqual(player_1.triple_rewards, [])
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [TabbyCat, AlleyCat, TabbyCat])
        self.assertEqual(player_1.in_play[1].golden, True)
        self.assertEqual(player_1.in_play[2].golden, True)
        self.assertEqual(player_1.in_play[1].attack, 2)
        self.assertEqual(player_1.in_play[2].attack, 2)
        self.assertEqual(player_1.in_play[1].health, 2)
        self.assertEqual(player_1.in_play[2].health, 2)
        self.assertEqual([card.level for card in player_1.triple_rewards], [2])

    def test_golden_token(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([AlleyCat] * 18)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        for _ in range(2):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            player_1.sell_board_minion(BoardIndex(len(player_1.in_play)-2))
            tavern.combat_step()

        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [AlleyCat])
        self.assertEqual(player_1.hand[0].golden, False)
        self.assertCardListEquals(player_1.in_play, [TabbyCat, TabbyCat])
        self.assertEqual(player_1.in_play[0].golden, False)
        self.assertEqual(player_1.in_play[1].golden, False)
        self.assertEqual(player_1.triple_rewards, [])
        player_1.summon_from_hand(HandIndex(0))

        self.assertCardListEquals(player_1.hand, [TabbyCat])
        self.assertEqual(player_1.hand[0].golden, True)
        self.assertCardListEquals(player_1.in_play, [AlleyCat])
        self.assertEqual(player_1.in_play[0].golden, False)
        self.assertEqual(player_1.triple_rewards, [])
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual([card.level for card in player_1.triple_rewards], [2])

    def test_buffed_golden(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([RighteousProtector] * 18)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Pyramad())
        player_2 = tavern.add_player_with_hero("lucy")
        for _ in range(2):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        player_1.purchase(StoreIndex(0))
        print(f"Player 1's board is: {player_1.in_play}")
        print(f"Player 1's hand is: {player_1.hand}")
        self.assertCardListEquals(player_1.hand, [RighteousProtector])
        self.assertEqual(player_1.hand[0].golden, True)
        self.assertEqual(player_1.hand[0].attack, 2)
        self.assertEqual(player_1.hand[0].health, 6)
        self.assertCardListEquals(player_1.in_play, [])

    class TestDiscoverCardRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            return force_card(cards, WrathWeaver)

        def select_discover_card(self, discoverables: List[Card]) -> Card:
            return force_card(discoverables, FreedealingGambler)

    def test_discover_card(self):
        tavern = Tavern()
        tavern.randomizer = self.TestDiscoverCardRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        for _ in range(2):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual([card.level for card in player_1.triple_rewards], [2])
        player_1.play_triple_rewards()
        player_1.select_discover(player_1.discovered_cards[0])
        print(f"Player 1's hand is: {player_1.hand}")
        self.assertCardListEquals(player_1.hand, [FreedealingGambler])
        self.assertCardListEquals(player_1.discovered_cards, [])

    class TestDiscoverGoldenRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            if round_number < 4:
                return force_card(cards, WrathWeaver)
            else:
                return force_card(cards, FreedealingGambler)

        def select_discover_card(self, discoverables: List[Card]) -> Card:
            return force_card(discoverables, FreedealingGambler)

    def test_discover_golden_trigger(self):
        tavern = Tavern()
        tavern.randomizer = self.TestDiscoverGoldenRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        for _ in range(3):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            tavern.combat_step()
        tavern.buying_step()
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual([card.level for card in player_1.triple_rewards], [2])
        player_1.upgrade_tavern()
        player_2.upgrade_tavern()
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.play_triple_rewards()
        player_1.select_discover(player_1.discovered_cards[0])
        print(player_1.hand)
        self.assertCardListEquals(player_1.hand, [FreedealingGambler])
        self.assertEqual(player_1.hand[0].golden, True)
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual([card.level for card in player_1.triple_rewards], [3])

    def test_micro_machine(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([MicroMachine] * 18)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.in_play[0].attack, 1)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.in_play[0].attack, 2)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.in_play[0].attack, 3)

    def test_murloc_tidehunter_and_tidecaller(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([MurlocTidecaller, MurlocTidehunter, MurlocTidehunter] * 6)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.in_play[0].attack, 1)
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MurlocTidecaller, MurlocTidehunter, MurlocScout])
        self.assertEqual(player_1.in_play[0].attack, 3)

    def test_vulgar_homunculus(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([VulgarHomunculus] * 6)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.health, 38)

    def test_metaltooth_leaper(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([MechaRoo] * 18 + [MetaltoothLeaper] * 8)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        # Round 1
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        # Round 2
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        # Round 3
        tavern.buying_step()
        player_1.upgrade_tavern()
        player_2.upgrade_tavern()
        tavern.combat_step()
        # Round 4
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.in_play, [MechaRoo, MechaRoo])
        self.assertEqual(player_1.in_play[0].attack, 1)
        self.assertEqual(player_1.in_play[1].attack, 1)

        player_1.summon_from_hand(HandIndex(0))

        self.assertCardListEquals(player_1.in_play, [MechaRoo, MechaRoo, MetaltoothLeaper])
        self.assertEqual(player_1.in_play[0].attack, 3)
        self.assertEqual(player_1.in_play[1].attack, 3)
        self.assertEqual(player_1.in_play[2].attack, 3)

    def test_rabid_saurolisk(self):
        tavern = Tavern()
        tavern.randomizer = CardForcer([MechaRoo, AlleyCat, AlleyCat] * 6 + [RabidSaurolisk] * 8)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        # Round 1
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        # Round 2
        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        tavern.combat_step()
        # Round 3
        tavern.buying_step()
        player_1.upgrade_tavern()
        player_2.upgrade_tavern()
        tavern.combat_step()
        # Round 4
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(2))
        self.assertCardListEquals(player_1.in_play, [RabidSaurolisk])
        self.assertEqual(player_1.in_play[0].attack, 3)
        self.assertEqual(player_1.in_play[0].health, 2)
        player_1.summon_from_hand(HandIndex(1))
        self.assertCardListEquals(player_1.in_play, [RabidSaurolisk, AlleyCat, TabbyCat])
        self.assertEqual(player_1.in_play[0].attack, 3)
        self.assertEqual(player_1.in_play[0].health, 2)
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [RabidSaurolisk, AlleyCat, TabbyCat, MechaRoo])
        self.assertEqual(player_1.in_play[0].attack, 4)
        self.assertEqual(player_1.in_play[0].health, 3)

    def test_steward_of_time(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = CardForcer([StewardOfTime] * 18)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.sell_hand_minion(HandIndex(0))
        self.assertEqual(player_1.store[0].attack, 4)
        self.assertEqual(player_1.store[0].health, 5)
        player_2.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_2.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.sell_hand_minion(HandIndex(0))
        self.assertEqual(player_2.store[0].attack, 5)
        self.assertEqual(player_2.store[0].health, 6)

    def test_deck_swabbie(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = CardForcer([DeckSwabbie] * 12)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.upgrade_tavern()
        self.assertEqual(player_1.tavern_tier, 2)

    def test_1am_6_24(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = CardForcer([Scallywag, MurlocTidehunter, MicroMachine, FiendishServant, FiendishServant, DeckSwabbie]*2)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(2))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()

    def test_rockpool_hunter(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = CardForcer([MurlocTidehunter, RockpoolHunter, RockpoolHunter]*4)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        self.assertEqual(player_1.in_play[0].attack, 2)
        self.assertEqual(player_1.in_play[0].health, 1)
        self.assertEqual(player_1.in_play[1].attack, 1)
        self.assertEqual(player_1.in_play[1].health, 1)
        self.assertCardListEquals(player_1.in_play, [MurlocTidehunter, MurlocScout])
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [MurlocTidehunter, MurlocScout, RockpoolHunter])
        self.assertEqual(player_1.in_play[0].attack, 3)
        self.assertEqual(player_1.in_play[0].health, 2)
        self.assertEqual(player_1.in_play[1].attack, 1)
        self.assertEqual(player_1.in_play[1].health, 1)
        self.assertEqual(player_1.in_play[2].attack, 2)
        self.assertEqual(player_1.in_play[2].health, 3)

    def test_millificent_manastorm(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Millificent", MillificentManastorm())
        player_2 = tavern.add_player_with_hero("Ethan")
        tavern.randomizer = CardForcer([MechaRoo] * 6)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertEqual(player_1.hand[0].attack, player_1.hand[0].base_attack + 1)
        self.assertEqual(player_1.hand[0].health, player_1.hand[0].base_health + 1)

    def test_yogg_saron(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Yogg", YoggSaron())
        player_2 = tavern.add_player_with_hero("Saron")
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(len(player_1.hand), 1)
        self.assertEqual(len(player_1.store), 2)
        self.assertEqual(player_1.coins, 1)
        self.assertEqual(player_1.hand[0].attack, player_1.hand[0].base_attack + 1)
        self.assertEqual(player_1.hand[0].health, player_1.hand[0].base_health + 1)

    def test_patches_the_pirate(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Yogg", PatchesThePirate())
        player_2 = tavern.add_player_with_hero("Saron")
        tavern.randomizer = CardForcer([Scallywag] * 12)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(len(player_1.hand), 2)
        self.assertEqual(player_1.coins, 1)
        self.assertEqual(player_1.hand[1].monster_type, MONSTER_TYPES.PIRATE)
        self.assertEqual(player_1.hand[1].tier, 1)

    def test_freedealing_gambler(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = CardForcer([FreedealingGambler] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [FreedealingGambler])
        coins = player_1.coins
        player_1.sell_hand_minion(HandIndex(0))
        self.assertEqual(player_1.coins, coins + 3)

    def test_income_limit(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        self.upgrade_to_tier(tavern, 6)
        tavern.buying_step()
        self.assertEqual(player_1.coins, 10)

    def test_crystal_weaver(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        tavern.randomizer = CardForcer([FiendishServant] * (18+16))
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = CardForcer([CrystalWeaver]*10)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [FiendishServant, CrystalWeaver])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)


    def test_nathrezim_overseer(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Josh")
        player_2 = tavern.add_player_with_hero("Diana")
    #     TODO: IS THIS A TEST??

    class TestGoldGrubberRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            if round_number < 6:
                return force_card(cards, FiendishServant)
            else:
                return force_card(cards, Goldgrubber)

    def test_gold_grubber(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        tavern.randomizer = self.TestGoldGrubberRandomizer()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 4)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [FiendishServant, Goldgrubber])
        tavern.combat_step()
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 2)

    def test_pogo_hopper(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = CardForcer([PogoHopper] * 16)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.counted_cards[PogoHopper], 2)
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 2)


    class TestZoobotRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[Card], player_name: str, round_number: int) -> Card:
            if round_number == 0:
                return force_card(cards, MurlocTidehunter)
            elif round_number == 1:
                return force_card(cards, DragonspawnLieutenant)
            elif round_number in (2,3):
                return force_card(cards, AlleyCat)
            elif round_number == 4:
                return force_card(cards, Zoobot)

        def select_friendly_minion(self, friendly_minions: List[Card]) -> Card:
            minion_types = [type(card) for card in friendly_minions]
            if MurlocTidehunter in minion_types:
                return force_card(friendly_minions, MurlocTidehunter)
            elif AlleyCat in minion_types:
                return force_card(friendly_minions, AlleyCat)
            else:
                return friendly_minions[0]

    def test_zoobot(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        tavern.randomizer = self.TestZoobotRandomizer()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(2))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.upgrade_tavern()
        player_2.upgrade_tavern()
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MurlocTidehunter, MurlocScout, DragonspawnLieutenant, AlleyCat, TabbyCat, Zoobot])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 1)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 1)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 1)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health + 1)
        self.assertEqual(player_1.in_play[4].attack, player_1.in_play[4].base_attack)
        self.assertEqual(player_1.in_play[4].health, player_1.in_play[4].base_health)
        self.assertEqual(player_1.in_play[5].attack, player_1.in_play[5].base_attack)
        self.assertEqual(player_1.in_play[5].health, player_1.in_play[5].base_health)

    def test_bloodsail_cannoneer(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        tavern.randomizer = RepeatedCardForcer([Scallywag])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([MechaRoo])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([BloodsailCannoneer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Scallywag, MechaRoo, BloodsailCannoneer])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)

    def test_coldlight_seer(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Bolly")
        player_2 = tavern.add_player_with_hero("Jolly")
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([ColdlightSeer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MurlocTidecaller, ColdlightSeer])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_crowd_favorite(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Josh")
        player_2 = tavern.add_player_with_hero("Jacob")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([CrowdFavorite, PogoHopper, PogoHopper, PogoHopper])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [CrowdFavorite, PogoHopper])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_crystal_weaver2(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Josh")
        player_2 = tavern.add_player_with_hero("Jacob")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([Imprisoner, CrystalWeaver])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Imprisoner, CrystalWeaver])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_felfin_navigator(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Bolly")
        player_2 = tavern.add_player_with_hero("Jolly")
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([FelfinNavigator])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MurlocTidecaller, FelfinNavigator])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_houndmaster(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([Houndmaster])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertEqual(player_1.in_play[0].attack, 1)
        self.assertEqual(player_1.in_play[0].health, 1)
        self.assertEqual(player_1.in_play[1].attack, 1)
        self.assertEqual(player_1.in_play[1].health, 1)
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat])
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, Houndmaster])
        self.assertEqual(player_1.in_play[0].attack, 3)
        self.assertEqual(player_1.in_play[0].health, 3)
        self.assertEqual(player_1.in_play[0].taunt, True)
        self.assertEqual(player_1.in_play[1].attack, 1)
        self.assertEqual(player_1.in_play[1].health, 1)
        self.assertEqual(player_1.in_play[2].attack, 4)
        self.assertEqual(player_1.in_play[2].health, 3)

    def test_screwjank_clunker(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = CardForcer([MechaRoo]*16)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([ScrewjankClunker] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        self.assertEqual(player_1.in_play[0].attack, 1)
        self.assertEqual(player_1.in_play[0].health, 1)
        self.assertEqual(player_1.in_play[1].attack, 1)
        self.assertEqual(player_1.in_play[1].health, 1)
        self.assertCardListEquals(player_1.in_play, [MechaRoo, MechaRoo])
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [MechaRoo, MechaRoo, ScrewjankClunker])
        self.assertEqual(player_1.in_play[0].attack, 3)
        self.assertEqual(player_1.in_play[0].health, 3)
        self.assertEqual(player_1.in_play[1].attack, 1)
        self.assertEqual(player_1.in_play[1].health, 1)
        self.assertEqual(player_1.in_play[2].attack, 2)
        self.assertEqual(player_1.in_play[2].health, 5)

    def test_reduced_tavern_upgrade_cost(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.upgrade_tavern()
        self.assertEqual(player_1.coins, 0)

    def test_freeze_not_busted(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.freeze()
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller])
        tavern.buying_step()
        self.assertCardListEquals(player_1.store, [AlleyCat, AlleyCat, MurlocTidecaller])

if __name__ == '__main__':
    unittest.main()
