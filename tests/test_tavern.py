import unittest
from typing import Type

from hearthstone.simulator.core.adaptations import AdaptBuffs
from hearthstone.simulator.core.card_graveyard import *
from hearthstone.simulator.core.card_pool import *
from hearthstone.simulator.core.cards import MonsterCard, CardType
from hearthstone.simulator.core.hero_graveyard import *
from hearthstone.simulator.core.hero_pool import *
from hearthstone.simulator.core.player import StoreIndex, HandIndex, BoardIndex
from hearthstone.simulator.core.randomizer import DefaultRandomizer
from hearthstone.simulator.core.tavern import Tavern


def force_card(cards: List[MonsterCard], card_type) -> MonsterCard:
    return [card for card in cards if isinstance(card, card_type)][0]


class CardForcer(DefaultRandomizer):
    def __init__(self, forced_cards: List[CardType]):
        super().__init__()
        self.forced_cards = forced_cards

    def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
        next_card_type = self.forced_cards.pop(0)
        return force_card(cards, next_card_type)


class RepeatedCardForcer(DefaultRandomizer):
    def __init__(self, repeatedly_forced_cards: List[CardType]):
        super().__init__()
        self.repeatedly_forced_cards = repeatedly_forced_cards
        self.pointer = 0

    def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
        next_card_type = self.repeatedly_forced_cards[self.pointer]
        self.pointer = (self.pointer + 1) % len(self.repeatedly_forced_cards)
        return force_card(cards, next_card_type)


class CardTests(unittest.TestCase):
    def assertCardListEquals(self, cards, expected, msg=None):
        self.assertListEqual([type(card) for card in cards], expected, msg=msg)

    def test_default_cardlist(self):
        default_cardlist = PrintingPress.make_cards(MONSTER_TYPES.single_types())
        self.assertGreater(len(default_cardlist), 20)
        print(f"the length of the default cardlist is {len(default_cardlist)}.")

    def test_draw(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(len(player_1.store), 3)
        self.assertEqual(len(player_2.store), 3)
        self.assertNotIn(None, player_1.store)
        self.assertNotIn(None, player_2.store)

    def type_of_cards(self, cards: List[MonsterCard]) -> List[Type]:
        return list(map(type, cards))

    class TestGameRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
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
        tavern = Tavern(restrict_types=False)
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

        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
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

        def select_attack_target(self, defenders: List[MonsterCard]) -> MonsterCard:
            target = [card for card in defenders if type(card) is not RighteousProtector]
            if target:
                return target[0]
            else:
                return defenders[0]

        def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
            number_of_battles = len(players) // 2
            return list(zip(players[:number_of_battles], players[number_of_battles:]))

    def test_two_rounds(self):
        tavern = Tavern(restrict_types=False)
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
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            return [card for card in cards if type(card) is AlleyCat][0]

    def test_battlecry(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestBattlecryRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertListEqual([AlleyCat, TabbyCat], self.type_of_cards(player_1.in_play))

    class TestRaftWeaverRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number == 0:
                return [card for card in cards if type(card) is WrathWeaver][0]
            else:
                return [card for card in cards if type(card) is FiendishServant][0]

    def test_raft_weaver(self):  # TODO: rename
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = CardForcer([MicroMachine] * 6)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.coins, 0)
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(player_1.coins, 1)

    def test_pyramad(self):
        tavern = Tavern(restrict_types=False)
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
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            return [card for card in cards if type(card) is WrathWeaver][0]

    def test_lord_jaraxxus(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", PatchWerk())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(player_1.health, 55)
        self.assertEqual(player_2.health, 40)

    def test_nefarian(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = CardForcer([AlleyCat] * 18)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        for _ in range(2):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            player_1.sell_minion(BoardIndex(len(player_1.in_play) - 2))
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
        tavern = Tavern(restrict_types=False)
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
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            return force_card(cards, WrathWeaver)

        def select_discover_card(self, discoverables: List[MonsterCard]) -> MonsterCard:
            minion_types = [type(card) for card in discoverables]
            if FreedealingGambler in minion_types:
                return force_card(discoverables, FreedealingGambler)
            else:
                return discoverables[0]

    def test_discover_card(self):
        tavern = Tavern(restrict_types=False)
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
        player_1.select_discover(0)
        print(f"Player 1's hand is: {player_1.hand}")
        self.assertCardListEquals(player_1.hand, [FreedealingGambler])
        self.assertCardListEquals(player_1.discover_queue, [])

    class TestDiscoverGoldenRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number < 4:
                return force_card(cards, WrathWeaver)
            else:
                return force_card(cards, FreedealingGambler)

        def select_discover_card(self, discoverables: List[MonsterCard]) -> MonsterCard:
            minion_types = [type(card) for card in discoverables]
            if FreedealingGambler in minion_types:
                return force_card(discoverables, FreedealingGambler)
            else:
                return discoverables[0]

    def test_discover_golden_trigger(self):
        tavern = Tavern(restrict_types=False)
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
        player_1.select_discover(0)
        print(player_1.hand)
        self.assertCardListEquals(player_1.hand, [FreedealingGambler])
        self.assertEqual(player_1.hand[0].golden, True)
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual([card.level for card in player_1.triple_rewards], [3])

    def test_micro_machine(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = CardForcer([VulgarHomunculus] * 6)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.health, 38)

    def test_metaltooth_leaper(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        self.assertEqual(player_1.in_play[0].attack, 4)
        self.assertEqual(player_1.in_play[0].health, 2)
        player_1.summon_from_hand(HandIndex(1))
        self.assertCardListEquals(player_1.in_play, [RabidSaurolisk, AlleyCat, TabbyCat])
        self.assertEqual(player_1.in_play[0].attack, 4)
        self.assertEqual(player_1.in_play[0].health, 2)
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [RabidSaurolisk, AlleyCat, TabbyCat, MechaRoo])
        self.assertEqual(player_1.in_play[0].attack, 5)
        self.assertEqual(player_1.in_play[0].health, 3)

    def test_steward_of_time(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = CardForcer([StewardOfTime] * 18)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(player_1.store[0].attack, 4)
        self.assertEqual(player_1.store[0].health, 5)

    def test_deck_swabbie(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Millificent", MillificentManastorm())
        player_2 = tavern.add_player_with_hero("Ethan")
        tavern.randomizer = CardForcer([MechaRoo] * 6)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertEqual(player_1.hand[0].attack, player_1.hand[0].base_attack + 1)
        self.assertEqual(player_1.hand[0].health, player_1.hand[0].base_health + 1)

    def test_yogg_saron(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = CardForcer([FreedealingGambler] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [FreedealingGambler])
        coins = player_1.coins
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(player_1.coins, coins + 3)

    def test_income_limit(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        self.upgrade_to_tier(tavern, 6)
        tavern.buying_step()
        self.assertEqual(player_1.coins, 10)

    def test_crystal_weaver(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Josh")
        player_2 = tavern.add_player_with_hero("Diana")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = RepeatedCardForcer([VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([NathrezimOverseer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [VulgarHomunculus, NathrezimOverseer])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    class TestGoldGrubberRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number < 6:
                return force_card(cards, FiendishServant)
            else:
                return force_card(cards, Goldgrubber)

    def test_gold_grubber(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number == 0:
                return force_card(cards, MurlocTidehunter)
            elif round_number == 1:
                return force_card(cards, DragonspawnLieutenant)
            elif round_number in (2, 3):
                return force_card(cards, AlleyCat)
            elif round_number == 4:
                return force_card(cards, Zoobot)

        def select_friendly_minion(self, friendly_minions: List[MonsterCard]) -> MonsterCard:
            minion_types = [type(card) for card in friendly_minions]
            if MurlocTidehunter in minion_types:
                return force_card(friendly_minions, MurlocTidehunter)
            elif AlleyCat in minion_types:
                return force_card(friendly_minions, AlleyCat)
            else:
                return friendly_minions[0]

    def test_zoobot(self):
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
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
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.upgrade_tavern()
        self.assertEqual(player_1.coins, 0)

    def test_freeze_not_busted(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.freeze()
        for card in player_1.store:
            self.assertTrue(card.frozen)
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller])
        tavern.buying_step()
        self.assertCardListEquals(player_1.store, [AlleyCat, AlleyCat, MurlocTidecaller])
        for card in player_1.store:
            self.assertFalse(card.frozen)

    def test_pack_leader(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = CardForcer([PackLeader] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([RabidSaurolisk] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.in_play, [PackLeader])
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [PackLeader, RabidSaurolisk])
        self.assertEqual(player_1.in_play[0].attack, 2)
        self.assertEqual(player_1.in_play[0].health, 3)
        self.assertEqual(player_1.in_play[1].attack, 6)
        self.assertEqual(player_1.in_play[1].health, 2)

    def test_salty_looter(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = CardForcer([SaltyLooter] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([Scallywag] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.in_play, [SaltyLooter])
        self.assertEqual(player_1.in_play[0].attack, 4)
        self.assertEqual(player_1.in_play[0].health, 4)
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [SaltyLooter, Scallywag])
        self.assertEqual(player_1.in_play[0].attack, 5)
        self.assertEqual(player_1.in_play[0].health, 5)
        self.assertEqual(player_1.in_play[1].attack, 2)
        self.assertEqual(player_1.in_play[1].health, 1)

    def test_twilight_emissary(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = CardForcer([DragonspawnLieutenant] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([TwilightEmissary] * 8)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant])
        self.assertEqual(player_1.in_play[0].attack, 2)
        self.assertEqual(player_1.in_play[0].health, 3)
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant, TwilightEmissary])
        self.assertEqual(player_1.in_play[0].attack, 4)
        self.assertEqual(player_1.in_play[0].health, 5)
        self.assertEqual(player_1.in_play[1].attack, 4)
        self.assertEqual(player_1.in_play[1].health, 4)

    def test_khadgar(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = CardForcer([Khadgar, AlleyCat] * 5)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Khadgar, AlleyCat, TabbyCat, TabbyCat])

    def test_double_khadgar(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([Khadgar])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play,
                                  [Khadgar, Khadgar, AlleyCat, TabbyCat])
        self.assertCardListEquals(player_1.hand,
                                  [TabbyCat])
        self.assertTrue(player_1.hand[0].golden)

    def test_virmen_sensei(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([ScavengingHyena])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([VirmenSensei])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.in_play, [ScavengingHyena])
        self.assertEqual(player_1.in_play[0].attack, 2)
        self.assertEqual(player_1.in_play[0].health, 2)
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [ScavengingHyena, VirmenSensei])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, 4)
        self.assertEqual(player_1.in_play[1].health, 5)

    def test_defender_of_argus(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = CardForcer([AlleyCat, DefenderOfArgus] * 5)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), targets=[BoardIndex(0), BoardIndex(1)])
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, DefenderOfArgus])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertTrue(player_1.in_play[0].taunt)
        self.assertTrue(player_1.in_play[1].taunt)

    def test_argus_one_target(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = CardForcer([AlleyCat, DefenderOfArgus] * 5)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, DefenderOfArgus])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertTrue(player_1.in_play[0].taunt)

    class TestDancinDerylRandomizer(DefaultRandomizer):
        def select_from_store(self, store: List['MonsterCard']) -> 'MonsterCard':
            return store[0]

    def test_dancin_deryl(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestDancinDerylRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", DancinDeryl())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(player_1.store[0].attack, player_1.store[0].base_attack + 2)
        self.assertEqual(player_1.store[0].health, player_1.store[0].base_health + 2)
        for i in range(1, len(player_1.store)):
            self.assertEqual(player_1.store[i].attack, player_1.store[i].base_attack)
            self.assertEqual(player_1.store[i].health, player_1.store[i].base_health)

    class TestFungalmancerFlurglRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
            return force_card(cards, MurlocTidecaller)

        def select_add_to_store(self, cards: List['MonsterCard']) -> 'MonsterCard':
            return force_card(cards, RockpoolHunter)

    def test_fungalmancer_flurgl(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestFungalmancerFlurglRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", FungalmancerFlurgl())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MurlocTidecaller])
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(type(player_1.store[-1]), RockpoolHunter)

    def test_kaelthas_sunstrider(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", KaelthasSunstrider())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = CardForcer([DragonspawnLieutenant] * 6)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([ScavengingHyena] * 6)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([Scallywag] * 6)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([VulgarHomunculus] * 6)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [DragonspawnLieutenant, ScavengingHyena, Scallywag, VulgarHomunculus])
        self.assertEqual(player_1.hand[0].attack, player_1.hand[0].base_attack)
        self.assertEqual(player_1.hand[0].health, player_1.hand[0].base_health)
        self.assertEqual(player_1.hand[1].attack, player_1.hand[1].base_attack)
        self.assertEqual(player_1.hand[1].health, player_1.hand[1].base_health)
        self.assertEqual(player_1.hand[2].attack, player_1.hand[2].base_attack + 2)
        self.assertEqual(player_1.hand[2].health, player_1.hand[2].base_health + 2)
        self.assertEqual(player_1.hand[3].attack, player_1.hand[3].base_attack)
        self.assertEqual(player_1.hand[3].health, player_1.hand[3].base_health)

    def test_lich_bazhial(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", LichBazhial())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(player_1.gold_coins, 1)
        player_1.redeem_gold_coin()
        self.assertEqual(player_1.health, 38)
        self.assertEqual(player_1.coins, 4)
        self.assertEqual(player_1.gold_coins, 0)

    def test_skycapn_kragg(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", SkycapnKragg())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(player_1.coins, 8)

    def test_the_curator_amalgam(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheCurator())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = CardForcer([RockpoolHunter] * 6)
        tavern.buying_step()
        self.assertCardListEquals(player_1.in_play, [Amalgam])
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [Amalgam, RockpoolHunter])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)

    class TestTheRatKingRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number == 0:
                return force_card(cards, ScavengingHyena)
            elif round_number == 1:
                return force_card(cards, MechaRoo)
            elif round_number == 2:
                return force_card(cards, Scallywag)
            elif round_number == 3:
                return force_card(cards, DragonspawnLieutenant)
            elif round_number == 4:
                return force_card(cards, FiendishServant)
            elif round_number == 5:
                return force_card(cards, MurlocTidecaller)

        def select_monster_type(self, monster_types: List['MONSTER_TYPES'], round_number: int) -> 'MONSTER_TYPES':
            if round_number == 0:
                return MONSTER_TYPES.BEAST
            elif round_number == 1:
                return MONSTER_TYPES.MECH
            elif round_number == 2:
                return MONSTER_TYPES.PIRATE
            elif round_number == 3:
                return MONSTER_TYPES.DRAGON
            elif round_number == 4:
                return MONSTER_TYPES.DEMON
            elif round_number == 5:
                return MONSTER_TYPES.MURLOC

    def test_the_rat_king(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheRatKing())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = self.TestTheRatKingRandomizer()
        for _ in range(6):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            tavern.combat_step()
        self.assertCardListEquals(player_1.hand, [ScavengingHyena, MechaRoo, Scallywag, DragonspawnLieutenant, FiendishServant, MurlocTidecaller])
        for i in range(player_1.hand_size()):
            self.assertEqual(player_1.hand[i].attack, player_1.hand[i].base_attack + 1)
            self.assertEqual(player_1.hand[i].health, player_1.hand[i].base_health + 2)

    class TestYseraRandomizer(DefaultRandomizer):
        def select_add_to_store(self, cards: List['MonsterCard']) -> 'MonsterCard':
            return force_card(cards, DragonspawnLieutenant)

    def test_ysera(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestYseraRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", Ysera())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(len(player_1.store), 4)
        self.assertEqual(type(player_1.store[3]), DragonspawnLieutenant)

    def test_bartendotron(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Bartendotron())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.upgrade_tavern()
        self.assertEqual(player_1.tavern_tier, 2)

    def test_millhouse_manastorm(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", MillhouseManastorm())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertEqual(player_1.coins, 1)
        tavern.combat_step()
        tavern.buying_step()
        player_1.reroll_store()
        self.assertEqual(player_1.coins, 2)
        tavern.combat_step()
        tavern.buying_step()
        player_1.upgrade_tavern()
        self.assertEqual(player_1.tavern_tier, 2)
        self.assertEqual(player_1.coins, 1)

    class TestShifterZerusRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
            if round_number < 4:
                return force_card(cards, AlleyCat)
            return force_card(cards, ShifterZerus)

        def select_random_minion(self, cards: List['Type'], round_number: int) -> 'Type':
            if round_number == 5:
                return MurlocTidecaller
            elif round_number == 6:
                return SavannahHighmane
            elif round_number == 7:
                return VulgarHomunculus

    def test_shifter_zerus(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestShifterZerusRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [ShifterZerus])
        tavern.combat_step()
        tavern.buying_step()
        self.assertCardListEquals(player_1.hand, [MurlocTidecaller])
        tavern.combat_step()
        tavern.buying_step()
        self.assertCardListEquals(player_1.hand, [SavannahHighmane])
        tavern.combat_step()
        tavern.buying_step()
        self.assertCardListEquals(player_1.hand, [VulgarHomunculus])
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [VulgarHomunculus])
        self.assertEqual(player_1.health, 38)
        self.assertTrue(player_1.in_play[0].taunt)

    def test_strongshell_scavenger(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([MechaRoo])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([StrongshellScavenger, AlleyCat] * 5)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant, MechaRoo, StrongshellScavenger])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)

    def test_annihilan_battlemaster(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([VulgarHomunculus, AnnihilanBattlemaster])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [VulgarHomunculus, AnnihilanBattlemaster])
        self.assertEqual(player_1.health, 38)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 2)

    def test_capn_hoggarr(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([CapnHoggarr, Scallywag])
        tavern.buying_step()
        gold = player_1.coins
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.in_play, [CapnHoggarr])
        self.assertCardListEquals(player_1.hand, [Scallywag, CapnHoggarr])
        self.assertEqual(player_1.coins, gold - 7)

    def test_king_bagurgle(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([MurlocTidehunter, DragonspawnLieutenant, KingBagurgle])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MurlocTidehunter, MurlocScout, DragonspawnLieutenant, KingBagurgle])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 2)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health)

    def test_razorgore_the_untamed(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([MurlocTidehunter, DragonspawnLieutenant, RazorgoreTheUntamed])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [MurlocTidehunter, MurlocScout, DragonspawnLieutenant, RazorgoreTheUntamed])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 2)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health + 2)

    def test_kalecgos_arcane_aspect(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant, KalecgosArcaneAspect])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([MechaRoo, RockpoolHunter])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant, KalecgosArcaneAspect, MechaRoo, RockpoolHunter])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health)

    def test_toxfin(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([RockpoolHunter, Toxfin])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [RockpoolHunter, Toxfin])
        self.assertTrue(player_1.in_play[0].poisonous)

    def test_floating_watcher(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([FloatingWatcher, VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [FloatingWatcher, VulgarHomunculus])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.health, 38)

    def test_mal_ganis(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([MalGanis, VulgarHomunculus, VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MalGanis, VulgarHomunculus])
        self.assertEqual(player_1.health, 40)
        player_1.sell_minion(BoardIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [VulgarHomunculus, VulgarHomunculus])
        self.assertEqual(player_1.health, 38)

    def test_mal_ganis_bugfix(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([MalGanis, VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(1))
        player_1.purchase(StoreIndex(1))
        player_1.purchase(StoreIndex(2))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [VulgarHomunculus])
        self.assertCardListEquals(player_1.hand, [MalGanis])
        self.assertTrue(player_1.hand[0].golden)
        self.assertEqual(player_1.health, 38)

    def test_mama_bear(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([MamaBear, AlleyCat, AlleyCat, AlleyCat, AlleyCat, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MamaBear, AlleyCat, TabbyCat])
        self.assertEqual(player_1.in_play[0].health, 4)
        self.assertEqual(player_1.in_play[0].attack, 4)
        self.assertEqual(player_1.in_play[1].health, 5)
        self.assertEqual(player_1.in_play[1].attack, 5)
        self.assertEqual(player_1.in_play[2].health, 5)
        self.assertEqual(player_1.in_play[2].attack, 5)

    def test_replicating_menace_magnetic(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([MicroMachine, ReplicatingMenace])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [MicroMachine])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(len(player_1.in_play[0].deathrattles), 1)

    def test_brann_bronzebeard(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([BrannBronzebeard, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [BrannBronzebeard, AlleyCat, TabbyCat, TabbyCat])

    def test_iron_sensei(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([AlleyCat, MechaRoo])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([IronSensei])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, MechaRoo, IronSensei])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 2)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 2)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health)

    class TestCaptainEudoraRandomizer(DefaultRandomizer):
        def select_gain_card(self, cards: List['MonsterCard']) -> 'MonsterCard':
            return force_card(cards, Goldgrubber)

    def test_captain_eudora(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestCaptainEudoraRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", CaptainEudora())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.buying_step()
        player_1.hero_power()
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        self.assertCardListEquals(player_1.hand, [Goldgrubber])
        self.assertTrue(player_1.hand[0].golden)

    class TestHangryDragonRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if player_name == "Dante_Kong":
                return force_card(cards, HangryDragon)
            if player_name == "lucy":
                return force_card(cards, FreedealingGambler)

    def test_hangry_dragon(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = self.TestHangryDragonRandomizer()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [HangryDragon])
        self.assertCardListEquals(player_2.in_play, [FreedealingGambler])
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_2.in_play, [FreedealingGambler, FreedealingGambler])
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)

    class TestLightfangEnforcerRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number == 6:
                return force_card(cards, MurlocTidecaller)
            elif round_number == 7:
                return force_card(cards, DragonspawnLieutenant)
            elif round_number == 8:
                return force_card(cards, VulgarHomunculus)
            elif round_number == 9:
                return force_card(cards, ScavengingHyena)
            elif round_number == 10:
                return force_card(cards, MechaRoo)
            else:
                return cards[0]

        def select_friendly_minion(self, friendly_minions: List[MonsterCard]) -> MonsterCard:
            minion_types = [type(card) for card in friendly_minions]
            if MurlocTidecaller in minion_types:
                return force_card(friendly_minions, MurlocTidecaller)
            elif DragonspawnLieutenant in minion_types:
                return force_card(friendly_minions, DragonspawnLieutenant)
            elif VulgarHomunculus in minion_types:
                return force_card(friendly_minions, VulgarHomunculus)
            elif ScavengingHyena in minion_types:
                return force_card(friendly_minions, ScavengingHyena)
            elif MechaRoo in minion_types:
                return force_card(friendly_minions, MechaRoo)
            else:
                return friendly_minions[0]

    def test_lightfang_enforcer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe", TheCurator())
        player_2 = tavern.add_player_with_hero("Donald")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = self.TestLightfangEnforcerRandomizer()
        for _ in range(5):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([LightfangEnforcer, AlleyCat])
        tavern.buying_step()
        tavern.randomizer = self.TestLightfangEnforcerRandomizer()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [Amalgam, MurlocTidecaller, DragonspawnLieutenant, VulgarHomunculus,
                                                     ScavengingHyena, MechaRoo, LightfangEnforcer])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 2)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 1)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 2)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health + 1)
        self.assertEqual(player_1.in_play[4].attack, player_1.in_play[4].base_attack + 2)
        self.assertEqual(player_1.in_play[4].health, player_1.in_play[4].base_health + 1)
        self.assertEqual(player_1.in_play[5].attack, player_1.in_play[5].base_attack + 2)
        self.assertEqual(player_1.in_play[5].health, player_1.in_play[5].base_health + 1)
        self.assertEqual(player_1.in_play[6].attack, player_1.in_play[6].base_attack)
        self.assertEqual(player_1.in_play[6].health, player_1.in_play[6].base_health)

    class TestMenagerieRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number == 0:
                return force_card(cards, MurlocTidehunter)
            elif round_number == 1:
                return force_card(cards, FiendishServant)
            elif round_number == 2:
                return force_card(cards, DragonspawnLieutenant)
            elif round_number == 3:
                return force_card(cards, MechaRoo)
            else:
                return force_card(cards, AlleyCat)

        def select_friendly_minion(self, friendly_minions: List[MonsterCard]) -> MonsterCard:
            minion_types = [type(card) for card in friendly_minions]
            if MurlocTidehunter in minion_types:
                return force_card(friendly_minions, MurlocTidehunter)
            elif FiendishServant in minion_types:
                return force_card(friendly_minions, FiendishServant)
            elif DragonspawnLieutenant in minion_types:
                return force_card(friendly_minions, DragonspawnLieutenant)
            elif AlleyCat in minion_types:
                return force_card(friendly_minions, AlleyCat)
            else:
                return friendly_minions[0]

    def test_menagerie_mug(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe", TheCurator())
        player_2 = tavern.add_player_with_hero("Donald")
        tavern.randomizer = self.TestMenagerieRandomizer()
        for _ in range(4):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = RepeatedCardForcer([MenagerieMug])
        tavern.buying_step()
        tavern.randomizer = self.TestMenagerieRandomizer()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [Amalgam, MurlocTidehunter, MurlocScout, FiendishServant,
                                                     DragonspawnLieutenant, MechaRoo, MenagerieMug])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 1)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health + 1)
        self.assertEqual(player_1.in_play[4].attack, player_1.in_play[4].base_attack + 1)
        self.assertEqual(player_1.in_play[4].health, player_1.in_play[4].base_health + 1)
        self.assertEqual(player_1.in_play[5].attack, player_1.in_play[5].base_attack)
        self.assertEqual(player_1.in_play[5].health, player_1.in_play[5].base_health)
        self.assertEqual(player_1.in_play[6].attack, player_1.in_play[6].base_attack)
        self.assertEqual(player_1.in_play[6].health, player_1.in_play[6].base_health)

    def test_menagerie_jug(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe", TheCurator())
        player_2 = tavern.add_player_with_hero("Donald")
        tavern.randomizer = self.TestMenagerieRandomizer()
        for _ in range(4):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([MenagerieJug])
        tavern.buying_step()
        tavern.randomizer = self.TestMenagerieRandomizer()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [Amalgam, MurlocTidehunter, MurlocScout, FiendishServant,
                                                     DragonspawnLieutenant, MechaRoo, MenagerieJug])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 2)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 2)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health + 2)
        self.assertEqual(player_1.in_play[4].attack, player_1.in_play[4].base_attack + 2)
        self.assertEqual(player_1.in_play[4].health, player_1.in_play[4].base_health + 2)
        self.assertEqual(player_1.in_play[5].attack, player_1.in_play[5].base_attack)
        self.assertEqual(player_1.in_play[5].health, player_1.in_play[5].base_health)
        self.assertEqual(player_1.in_play[6].attack, player_1.in_play[6].base_attack)
        self.assertEqual(player_1.in_play[6].health, player_1.in_play[6].base_health)

    def test_queen_wagtoggle(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe", QueenWagtoggle())
        player_2 = tavern.add_player_with_hero("Donald")
        tavern.randomizer = self.TestMenagerieRandomizer()
        for _ in range(5):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        self.assertCardListEquals(player_1.in_play, [MurlocTidehunter, MurlocScout, FiendishServant,
                                                     DragonspawnLieutenant, MechaRoo, AlleyCat, TabbyCat])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 2)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 2)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health)
        self.assertEqual(player_1.in_play[4].attack, player_1.in_play[4].base_attack + 2)
        self.assertEqual(player_1.in_play[4].health, player_1.in_play[4].base_health)
        self.assertEqual(player_1.in_play[5].attack, player_1.in_play[5].base_attack + 2)
        self.assertEqual(player_1.in_play[5].health, player_1.in_play[5].base_health)
        self.assertEqual(player_1.in_play[6].attack, player_1.in_play[6].base_attack)
        self.assertEqual(player_1.in_play[6].health, player_1.in_play[6].base_health)

    class TestMicroMummyRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number == 0:
                return force_card(cards, AlleyCat)
            else:
                return force_card(cards, MicroMummy)

        def select_friendly_minion(self, friendly_minions: List[MonsterCard]) -> MonsterCard:
            minion_types = [type(card) for card in friendly_minions]
            if AlleyCat in minion_types:
                return force_card(friendly_minions, AlleyCat)
            else:
                return friendly_minions[0]

    def test_micro_mummy(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = self.TestMicroMummyRandomizer()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, MicroMummy])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)

    def test_forest_warden_omu(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", ForestWardenOmu())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.upgrade_tavern()
        self.assertEqual(player_1.coins, 2)

    def test_george_the_fallen(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", GeorgeTheFallen())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertTrue(player_1.in_play[0].divine_shield)
        self.assertFalse(player_1.in_play[1].divine_shield)

    def test_reno_jackson(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", RenoJackson())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertTrue(player_1.in_play[0].golden)
        self.assertFalse(player_1.in_play[1].golden)

    class TestJandiceBarovRandomizer(DefaultRandomizer):
        def select_from_store(self, store: List['MonsterCard']) -> 'MonsterCard':
            minion_types = [type(card) for card in store]
            if VulgarHomunculus in minion_types:
                return force_card(store, VulgarHomunculus)
            else:
                return store[0]

    def test_jandice_barov(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", JandiceBarov())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat, AlleyCat, VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.randomizer = self.TestJandiceBarovRandomizer()
        player_1.hero_power(board_index=BoardIndex(1))
        self.assertCardListEquals(player_1.in_play, [AlleyCat, VulgarHomunculus])
        self.assertCardListEquals(player_1.store, [AlleyCat, TabbyCat])

    class TestAmalgadonRandomizer(DefaultRandomizer):
        def select_adaptation(self, adaptation_types: List['Type']) -> 'Type':
            if AdaptBuffs.CracklingShield in adaptation_types:
                return AdaptBuffs.CracklingShield
            if AdaptBuffs.LivingSpores in adaptation_types:
                return AdaptBuffs.LivingSpores
            else:
                return adaptation_types[0]

    def test_amalgadon(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheCurator())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([Amalgadon, AlleyCat, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.randomizer = self.TestAmalgadonRandomizer()
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Amalgam, Amalgadon])
        self.assertTrue(player_1.in_play[1].divine_shield)
        self.assertEqual(len(player_1.in_play[1].deathrattles), 1)

    def test_arch_villain_rafaam(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", ArchVillianRafaam())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie, ScavengingHyena, ScavengingHyena])
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(1))
        player_2.summon_from_hand(HandIndex(0))
        player_1.hero_power()
        self.assertCardListEquals(player_1.in_play, [DeckSwabbie])
        self.assertCardListEquals(player_2.in_play, [ScavengingHyena])
        tavern.combat_step()
        self.assertCardListEquals(player_1.hand, [ScavengingHyena])

    def test_annoy_o_module(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([PogoHopper, AnnoyOModule, AnnoyOModule])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [PogoHopper])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 4)
        self.assertTrue(player_1.in_play[0].taunt)
        self.assertTrue(player_1.in_play[0].divine_shield)

    def test_not_magnetized(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([AnnoyOModule, ReplicatingMenace])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [AnnoyOModule, ReplicatingMenace])

    def test_golden_module(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([PogoHopper])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([AnnoyOModule])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [AnnoyOModule])
        self.assertTrue(player_1.hand[0].golden)
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [PogoHopper])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 4)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 8)
        self.assertTrue(player_1.in_play[0].taunt)
        self.assertTrue(player_1.in_play[0].divine_shield)
        self.assertCardListEquals(player_1.in_play[0].attached_cards, [AnnoyOModule])
        self.assertTrue(player_1.in_play[0].attached_cards[0].golden)

    def test_golden_menace(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([PogoHopper, KalecgosArcaneAspect])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(1))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([ReplicatingMenace])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [PogoHopper, ReplicatingMenace])
        self.assertCardListEquals(player_2.hand, [KalecgosArcaneAspect])
        self.assertTrue(player_1.hand[1].golden)
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [PogoHopper])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 6)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(len(player_1.in_play[0].deathrattles), 1)
        self.assertCardListEquals(player_1.in_play[0].attached_cards, [ReplicatingMenace])
        self.assertTrue(player_1.in_play[0].attached_cards[0].golden)
        self.assertCardListEquals(player_2.in_play, [KalecgosArcaneAspect])
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 40)

    def test_magnetized_menace_wont_turn_golden(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([PogoHopper, ReplicatingMenace])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [PogoHopper])
        self.assertCardListEquals(player_1.in_play[0].attached_cards, [ReplicatingMenace])
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([ReplicatingMenace])
        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        player_1.purchase(StoreIndex(1))
        self.assertCardListEquals(player_1.hand, [ReplicatingMenace, ReplicatingMenace])
        self.assertCardListEquals(player_1.in_play[0].attached_cards, [ReplicatingMenace])

    def test_rafaam_whelp(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", ArchVillianRafaam())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([RedWhelp, RedWhelp, WrathWeaver])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(2))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power()
        self.assertCardListEquals(player_1.in_play, [RedWhelp, RedWhelp])
        self.assertCardListEquals(player_2.in_play, [WrathWeaver])
        tavern.combat_step()
        self.assertCardListEquals(player_1.hand, [WrathWeaver])

    class TestCobaltScalebaneRandomizer(DefaultRandomizer):
        def select_friendly_minion(self, friendly_minions: List['MonsterCard']) -> 'MonsterCard':
            minion_types = [type(card) for card in friendly_minions]
            if AlleyCat in minion_types:
                return force_card(friendly_minions, AlleyCat)
            else:
                return friendly_minions[0]

    def test_cobalt_scalebane(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([CobaltScalebane, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [CobaltScalebane, AlleyCat, TabbyCat])
        tavern.randomizer = self.TestCobaltScalebaneRandomizer()
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 3)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)

    def test_dissolve_magnetic(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([PogoHopper, ReplicatingMenace])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        print(player_1.in_play[0].attached_cards)
        player_1.sell_minion(BoardIndex(0))

    def test_captain_hooktusk(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", CaptainHooktusk())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([CrystalWeaver])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertEqual(player_1.hand_size(), 1)
        self.assertLess(player_1.hand[0].tier, 3)
        self.assertEqual(len(player_1.in_play), 0)

    def test_hooktusk_tier_one_minion(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", CaptainHooktusk())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertEqual(player_1.hand_size(), 1)
        self.assertEqual(player_1.hand[0].tier, 1)
        self.assertEqual(len(player_1.in_play), 0)

    def test_malygos(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Malygos())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        noted_tier = player_1.in_play[0].tier
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].tier, noted_tier)

    def test_AFKay(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", AFKay())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(player_1.coins, 0)
        self.assertEqual(len(player_1.triple_rewards), 0)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.coins, 0)
        self.assertEqual(len(player_1.triple_rewards), 0)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(len(player_1.triple_rewards), 2)
        for reward in player_1.triple_rewards:
            self.assertEqual(reward.level, 3)

    def test_southsea_strongarm(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie, SouthseaStrongarm])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [DeckSwabbie, SouthseaStrongarm])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_edwin_vancleef(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", EdwinVanCleef())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie, DeckSwabbie])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(1))
        self.assertCardListEquals(player_1.in_play, [DeckSwabbie, DeckSwabbie])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 2)

    def test_primalfin_lookout(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller, PrimalfinLookout])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MurlocTidecaller, PrimalfinLookout])
        self.assertEqual(len(player_1.discover_queue[0]), 3)
        for card in player_1.discover_queue[0]:
            self.assertTrue(card.check_type(MONSTER_TYPES.MURLOC))

    def test_golden_primalfin(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller, PrimalfinLookout])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([PrimalfinLookout])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MurlocTidecaller, PrimalfinLookout])
        self.assertTrue(player_1.in_play[1].golden)
        self.assertEqual(len(player_1.discover_queue), 2)
        player_1.play_triple_rewards()
        self.assertEqual(len(player_1.discover_queue), 3)
        for _ in range(3):
            player_1.select_discover(0)
        self.assertEqual(len(player_1.hand), 3)
        self.assertEqual(len(player_1.discover_queue), 0)

    def test_murozond(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant, Goldgrubber])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([Murozond, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant, Murozond])
        self.assertCardListEquals(player_1.hand, [Goldgrubber])

    def test_golden_murozond(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant, Murozond, Goldgrubber])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([Murozond, Murozond, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant, Murozond])
        self.assertTrue(player_1.in_play[1].golden)
        self.assertCardListEquals(player_1.hand, [Goldgrubber])
        self.assertTrue(player_1.hand[0].golden)

    def test_aranna_starseeker(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", ArannaStarseeker())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        for _ in range(3):
            player_1.reroll_store()
        tavern.combat_step()
        tavern.buying_step()
        for _ in range(2):
            player_1.reroll_store()
        self.assertEqual(7, len(player_1.store))

    class DinotamerBrannRandomizer(DefaultRandomizer):

        def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
            for card in cards:
                assert card.base_battlecry
            return self.rand.choice(cards)

    def test_dinotamer_brann(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", DinotamerBrann())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        tavern.randomizer = self.DinotamerBrannRandomizer()
        player_1.hero_power()

    def test_alexstrasza(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Alexstrasza())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        self.assertEqual(len(player_1.discover_queue), 2)
        for _ in range(2):
            self.assertTrue(card.check_type(MONSTER_TYPES.DRAGON) for card in player_1.discover_queue[0])
            player_1.select_discover(0)
        self.assertEqual(len(player_1.hand), 2)
        self.assertEqual(len(player_1.discover_queue), 0)

    def test_king_mukla(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", KingMukla())
        player_2 = tavern.add_player_with_hero("lucy")
        player_3 = tavern.add_player_with_hero("thing_1")
        player_4 = tavern.add_player_with_hero("thing_2")
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(player_1.bananas, 2)
        self.assertEqual(player_2.bananas, 0)
        self.assertEqual(player_3.bananas, 0)
        self.assertEqual(player_4.bananas, 0)
        tavern.combat_step()
        self.assertEqual(player_1.bananas, 2)
        self.assertEqual(player_2.bananas, 1)
        self.assertEqual(player_3.bananas, 1)
        self.assertEqual(player_4.bananas, 1)
        tavern.buying_step()
        player_1.use_banana(store_index=StoreIndex(0))
        player_1.use_banana(store_index=StoreIndex(0))
        self.assertEqual(player_1.store[0].attack, player_1.store[0].base_attack + 2)
        self.assertEqual(player_1.store[0].health, player_1.store[0].base_health + 2)

    def test_elise_starseeker(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", EliseStarseeker())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 2)
        self.assertEqual(len(player_1.hero.recruitment_maps), 1)
        self.assertEqual(player_1.hero.recruitment_maps[0], 2)
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(len(player_1.discover_queue), 1)
        player_1.select_discover(0)
        self.assertEqual(len(player_1.hand), 1)
        self.assertEqual(len(player_1.discover_queue), 0)

    def test_party_elemental(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([PartyElemental, CracklingCyclone])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [PartyElemental, CracklingCyclone])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)

    def test_molten_rock(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([MoltenRock, CracklingCyclone])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MoltenRock, CracklingCyclone])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_arcane_assistant(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([CracklingCyclone, ArcaneAssistant])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [CracklingCyclone, ArcaneAssistant])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_lieutenant_garr(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([LieutenantGarr, CracklingCyclone])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [LieutenantGarr, CracklingCyclone])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_al_akir(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", AlAkir())
        player_2 = tavern.add_player_with_hero("lucy")
        for _ in range(3):
            tavern.buying_step()
            tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie, AlleyCat, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [DeckSwabbie])
        self.assertCardListEquals(player_2.in_play, [DeckSwabbie, AlleyCat, TabbyCat])
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 40)

    def test_sellemental(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([Sellemental])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Sellemental])
        player_1.sell_minion(BoardIndex(0))
        self.assertCardListEquals(player_1.hand, [WaterDroplet])

    def test_chenvaala(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Chenvaala())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([Sellemental])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Sellemental, WaterDroplet])
        self.assertEqual(player_1.tavern_upgrade_cost, 1)

    def test_ragnaros_the_firelord(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", RagnarosTheFirelord())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat])
        self.assertCardListEquals(player_2.in_play, [AlleyCat, TabbyCat])
        for _ in range(13):
            self.assertFalse(player_1.hero.sulfuras)
            tavern.combat_step()
            tavern.buying_step()
        self.assertTrue(player_1.hero.sulfuras)
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 3)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 3)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 3)

    def test_rakanishu(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Rakanishu())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([Sellemental])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power()
        self.assertCardListEquals(player_1.in_play, [Sellemental])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 6)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 6)

    def test_mr_bigglesworth(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", MrBigglesworth())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([AlleyCat, DefenderOfArgus, ZappSlywick])
        tavern.buying_step()
        player_1.purchase(StoreIndex(2))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0), [BoardIndex(0), BoardIndex(1)])
        self.assertCardListEquals(player_1.in_play, [ZappSlywick])
        self.assertCardListEquals(player_2.in_play, [AlleyCat, TabbyCat, DefenderOfArgus])
        self.assertEqual(player_2.in_play[0].attack, player_2.in_play[0].base_attack + 1)
        self.assertEqual(player_2.in_play[0].health, player_2.in_play[0].base_health + 1)
        self.assertEqual(player_2.in_play[1].attack, player_2.in_play[1].base_attack + 1)
        self.assertEqual(player_2.in_play[1].health, player_2.in_play[1].base_health + 1)
        self.assertTrue(player_2.in_play[0].taunt)
        self.assertTrue(player_2.in_play[1].taunt)
        for _ in range(4):
            self.assertFalse(player_2.dead)
            tavern.combat_step()
            tavern.buying_step()
        self.assertTrue(player_2.dead)
        self.assertFalse(bool(player_2.in_play))
        self.assertEqual(len(player_1.discover_queue), 1)
        self.assertEqual(len(player_1.discover_queue[0]), 3)
        for card in player_1.discover_queue[0]:
            if type(card) != DefenderOfArgus:
                self.assertEqual(card.attack, card.base_attack + 1)
                self.assertEqual(card.health, card.base_health + 1)
                self.assertTrue(card.taunt)
        player_1.select_discover(0)
        self.assertEqual(len(player_1.hand), 1)
        self.assertEqual(len(player_1.discover_queue), 0)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(len(player_1.discover_queue), 0)

    def test_chenvaala_stop_at_zero(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Chenvaala())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([Sellemental])
        for _ in range(4):
            tavern.buying_step()
            tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.tavern_upgrade_cost, 1)
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Sellemental, WaterDroplet])
        self.assertEqual(player_1.tavern_upgrade_cost, 0)

    def test_ragnaros_one_minion(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", RagnarosTheFirelord())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat])
        self.assertCardListEquals(player_2.in_play, [AlleyCat, TabbyCat])
        for _ in range(13):
            self.assertFalse(player_1.hero.sulfuras)
            tavern.combat_step()
            tavern.buying_step()
        self.assertTrue(player_1.hero.sulfuras)
        player_1.sell_minion(BoardIndex(1))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 6)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 6)

    def test_triple_water_droplet(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([Sellemental])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [WaterDroplet, Sellemental])
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(1))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(1))
        self.assertCardListEquals(player_1.in_play, [WaterDroplet, Sellemental, Sellemental])
        self.assertCardListEquals(player_1.hand, [WaterDroplet])
        player_1.sell_minion(BoardIndex(1))
        self.assertCardListEquals(player_1.in_play, [Sellemental])
        self.assertCardListEquals(player_1.hand, [WaterDroplet])
        self.assertTrue(player_1.hand[0].golden)

    class TestLilRagRandomizer(DefaultRandomizer):
        def select_friendly_minion(self, friendly_minions: List['MonsterCard']) -> 'MonsterCard':
            minion_types = [type(card) for card in friendly_minions]
            if AlleyCat in minion_types:
                return force_card(friendly_minions, AlleyCat)
            else:
                return friendly_minions[0]

    def test_lil_rag(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([AlleyCat, LilRag, Sellemental])
        tavern.buying_step()
        for _ in range(3):
            player_1.purchase(StoreIndex(0))
        for _ in range(2):
            player_1.summon_from_hand(HandIndex(0))
        tavern.randomizer = self.TestLilRagRandomizer()
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, LilRag, Sellemental])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health)

    def test_tavern_tempest(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([TavernTempest, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.hand_size(), 1)
        self.assertTrue(player_1.hand[0].check_type(MONSTER_TYPES.ELEMENTAL) and player_1.hand[0].tier <= player_1.tavern_tier)

    def test_nomi_kitchen_nightmare(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([NomiKitchenNightmare, Sellemental, Sellemental])
        tavern.buying_step()
        for _ in range(3):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [NomiKitchenNightmare, Sellemental, Sellemental])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 1)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 1)

    def test_nomi_jandice(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", JandiceBarov())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([NomiKitchenNightmare, Sellemental, Sellemental, Sellemental, Sellemental])
        tavern.buying_step()
        for _ in range(2):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [NomiKitchenNightmare, Sellemental])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        player_1.hero_power(board_index=BoardIndex(1))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)
        player_1.purchase(StoreIndex(2))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [NomiKitchenNightmare, Sellemental, Sellemental])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 1)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 1)

    def test_excluded_types(self):
        tavern = Tavern()
        for hero in tavern.hero_pool:
            self.assertTrue(hero.pool in tavern.available_types or hero.pool == MONSTER_TYPES.ALL)
        for card in tavern.deck.all_cards():
            self.assertTrue(card.pool in tavern.available_types or card.pool == MONSTER_TYPES.ALL)

    def test_refreshing_anomaly(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([RefreshingAnomaly])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertEqual(player_1.coins, 0)
        self.assertEqual(player_1.free_refreshes, 0)
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.free_refreshes, 1)
        player_1.reroll_store()
        self.assertEqual(player_1.free_refreshes, 0)

    def test_two_refreshing_anomalies(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([RefreshingAnomaly])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.free_refreshes, 1)
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.free_refreshes, 1)
        player_1.reroll_store()
        self.assertEqual(player_1.free_refreshes, 0)
        self.assertEqual(player_1.coins, 1)

    def test_golden_refreshing_anomaly(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([RefreshingAnomaly])
        for _ in range(3):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            tavern.combat_step()
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.coins, 2)
        self.assertEqual(player_1.free_refreshes, 2)
        player_1.reroll_store()
        self.assertEqual(player_1.coins, 2)
        self.assertEqual(player_1.free_refreshes, 1)
        player_1.reroll_store()
        self.assertEqual(player_1.coins, 2)
        self.assertEqual(player_1.free_refreshes, 0)

    def test_nozdormu(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Nozdormu())
        player_2 = tavern.add_player_with_hero("lucy")
        self.assertEqual(player_1.free_refreshes, 0)
        tavern.buying_step()
        self.assertEqual(player_1.free_refreshes, 1)
        player_1.reroll_store()
        self.assertEqual(player_1.coins, 3)
        self.assertEqual(player_1.free_refreshes, 0)

    def test_nozdormu_refreshing_anomaly(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Nozdormu())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([RefreshingAnomaly])
        self.assertEqual(player_1.free_refreshes, 0)
        tavern.buying_step()
        self.assertEqual(player_1.free_refreshes, 1)
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand((HandIndex(0)))
        self.assertEqual(player_1.free_refreshes, 1)
        player_1.reroll_store()
        self.assertEqual(player_1.free_refreshes, 0)

    def test_majordomo_executus(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([Sellemental, MajordomoExecutus])
        tavern.buying_step()
        self.assertEqual(len(player_1.played_minions), 0)
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand((HandIndex(0)))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand((HandIndex(0)))
        self.assertEqual(len(player_1.played_minions), 2)
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    def test_frozen_interactions(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", JandiceBarov())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.freeze()
        player_1.purchase(StoreIndex(0))
        self.assertFalse(player_1.hand[0].frozen)
        self.assertTrue(player_1.store[0].frozen)
        self.assertTrue(player_1.store[1].frozen)
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.freeze()
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertFalse(player_1.in_play[0].frozen)
        self.assertTrue(player_1.store[0].frozen)
        self.assertTrue(player_1.store[1].frozen)
        self.assertFalse(player_1.store[2].frozen)

    def test_stasis_elemental(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([StasisElemental])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand((HandIndex(0)))
        self.assertEqual(len(player_1.store), 4)
        self.assertTrue(player_1.store[3].frozen and player_1.store[3].check_type(MONSTER_TYPES.ELEMENTAL))

    def test_sindragosa(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Sindragosa())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.freeze()
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.store[0].attack, player_1.store[0].base_attack + 2)
        self.assertEqual(player_1.store[0].health, player_1.store[0].base_health + 1)
        self.assertEqual(player_1.store[1].attack, player_1.store[1].base_attack + 2)
        self.assertEqual(player_1.store[1].health, player_1.store[1].base_health + 1)
        self.assertEqual(player_1.store[2].attack, player_1.store[2].base_attack + 2)
        self.assertEqual(player_1.store[2].health, player_1.store[2].base_health + 1)

    def test_galakrond(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Galakrond())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertFalse(player_1.store[0].frozen)
        self.assertFalse(player_1.store[1].frozen)
        self.assertTrue(player_1.store[2].frozen)
        self.assertEqual(player_1.store[2].tier, 2)
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([Amalgadon, AlleyCat])
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertFalse(player_1.store[0].frozen)
        self.assertFalse(player_1.store[1].frozen)
        self.assertFalse(player_1.store[2].frozen)
        self.assertFalse(player_1.store[3].frozen)
        self.assertFalse(player_1.store[4].frozen)
        self.assertTrue(player_1.store[5].frozen)
        self.assertEqual(player_1.store[5].tier, 6)

    def test_infinite_toki(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", InfiniteToki())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(player_1.store[0].tier, 1)
        self.assertEqual(player_1.store[1].tier, 1)
        self.assertEqual(player_1.store[2].tier, 2)
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 6)
        tavern.buying_step()
        player_1.hero_power()
        print(player_1.store)

    def test_frozen_reroll(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Galakrond())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertFalse(player_1.store[0].frozen)
        self.assertFalse(player_1.store[1].frozen)
        self.assertTrue(player_1.store[2].frozen)
        self.assertEqual(player_1.store[2].tier, 2)
        player_1.reroll_store()
        for card in player_1.store:
            self.assertEqual(card.tier, 1)
            self.assertFalse(card.frozen)

    def test_the_lich_king(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheLichKing())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([WrathWeaver, AlleyCat, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand((HandIndex(0)))
        player_2.purchase(StoreIndex(1))
        player_2.summon_from_hand((HandIndex(0)))
        self.assertCardListEquals(player_1.in_play, [WrathWeaver])
        self.assertCardListEquals(player_2.in_play, [AlleyCat, TabbyCat])
        player_1.hero_power(board_index=BoardIndex(0))
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 40)

    def test_golden_goliath_transformation(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([SeabreakerGoliath, AlleyCat])
        for _ in range(3):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand((HandIndex(0)))
            tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [SeabreakerGoliath])
        self.assertTrue(player_1.in_play[0].golden)
        self.assertTrue(player_1.in_play[0].mega_windfury)
        self.assertFalse(player_1.in_play[0].windfury)

    class TestGoldenAmalgadonRandomizer(DefaultRandomizer):
        def select_adaptation(self, adaptation_types: List['Type']) -> 'Type':
            if AdaptBuffs.LightningSpeed in adaptation_types:
                return AdaptBuffs.LightningSpeed
            else:
                return adaptation_types[0]

    def test_golden_amalgadon_windfury(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([Amalgadon, AlleyCat, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.randomizer = self.TestGoldenAmalgadonRandomizer()
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Amalgadon])
        self.assertTrue(player_1.in_play[0].windfury)
        self.assertFalse(player_1.in_play[0].mega_windfury)

    class TestTessGreymaneRandomizer(DefaultRandomizer):
        def select_add_to_store(self, cards: List['MonsterCard']) -> 'MonsterCard':
            minion_types = [type(card) for card in cards]
            if AlleyCat in minion_types:
                return force_card(cards, AlleyCat)
            elif MurlocTidehunter in minion_types:
                return force_card(cards, MurlocTidehunter)
            elif DragonspawnLieutenant in minion_types:
                return force_card(cards, DragonspawnLieutenant)
            else:
                return cards[0]

    def test_tess_greymane(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TessGreymane())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat, MurlocTidehunter, DragonspawnLieutenant])
        for i in range(3):
            tavern.buying_step()
            player_2.purchase(StoreIndex(i))
            player_2.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        self.assertCardListEquals(player_2.in_play, [AlleyCat, TabbyCat, MurlocTidehunter, MurlocScout, DragonspawnLieutenant])
        tavern.buying_step()
        tavern.randomizer = self.TestTessGreymaneRandomizer()
        player_1.hero_power()
        self.assertCardListEquals(player_1.store, [AlleyCat, MurlocTidehunter, DragonspawnLieutenant])

    def test_malygos_store_target(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Malygos())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.buying_step()
        noted_tier = player_1.store[0].tier
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertEqual(player_1.store[5].tier, noted_tier)

    def test_shudderwock(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Shudderwock())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat, MurlocTidehunter, VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, TabbyCat, MurlocTidehunter, MurlocScout])

    def test_brann_shudderwock_doesnt_stack(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Shudderwock())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([BrannBronzebeard, BrannBronzebeard, AlleyCat])
        tavern.buying_step()
        player_1.hero_power()
        for _ in range(3):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [BrannBronzebeard, BrannBronzebeard, AlleyCat, TabbyCat, TabbyCat])

    class TestZerusNotBuggedRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
            if round_number <= 3:
                return force_card(cards, AlleyCat)
            return force_card(cards, ShifterZerus)

        def select_random_minion(self, cards: List['Type'], round_number: int) -> 'Type':
            if round_number == 5:
                return AlleyCat
            elif round_number == 6:
                return DragonspawnLieutenant
            elif round_number == 7:
                return ShifterZerus
            else:
                return cards[0]

    def test_zerus_not_bugged(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestZerusNotBuggedRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [ShifterZerus])
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [AlleyCat, ShifterZerus])
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [DragonspawnLieutenant, DragonspawnLieutenant, ShifterZerus])
        tavern.combat_step()
        tavern.buying_step()
        self.assertCardListEquals(player_1.hand, [ShifterZerus])
        self.assertTrue(player_1.hand[0].golden)


if __name__ == '__main__':
    unittest.main()
