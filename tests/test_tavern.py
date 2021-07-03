import unittest

from hearthstone.simulator.agent.actions import generate_standard_actions, EndPhaseAction
from hearthstone.simulator.core.adaptations import AdaptBuffs
from hearthstone.simulator.core.card_graveyard import *
from hearthstone.simulator.core.card_pool import *
from hearthstone.simulator.core.cards import MonsterCard
from hearthstone.simulator.core.hero_graveyard import *
from hearthstone.simulator.core.hero_pool import *
from hearthstone.simulator.core.player import HandIndex, DiscoverIndex, Player, SpellIndex
from hearthstone.simulator.core.randomizer import DefaultRandomizer
from hearthstone.simulator.core.secrets import Secret
from hearthstone.simulator.core.spell_pool import *
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.testing.battlegrounds_test_case import BattleGroundsTestCase


def force_card(cards: List[MonsterCard], card_type) -> MonsterCard:
    return [card for card in cards if isinstance(card, card_type)][0]


class CardForcer(DefaultRandomizer):
    def __init__(self, forced_cards: List[Type['MonsterCard']]):
        super().__init__()
        self.forced_cards = forced_cards

    def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
        next_card_type = self.forced_cards.pop(0)
        return force_card(cards, next_card_type)


class RepeatedCardForcer(DefaultRandomizer):
    def __init__(self, repeatedly_forced_cards: List[Type['MonsterCard']]):
        super().__init__()
        self.repeatedly_forced_cards = repeatedly_forced_cards
        self.pointer = 0

    def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
        next_card_type = self.repeatedly_forced_cards[self.pointer]
        self.pointer = (self.pointer + 1) % len(self.repeatedly_forced_cards)
        return force_card(cards, next_card_type)


class CardTests(BattleGroundsTestCase):
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
                return force_card(cards, FiendishServant)
            if player_name == "lucy":
                return force_card(cards, Scallywag)

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
        self.assertCardListEquals(player_1.store, [FiendishServant, FiendishServant, FiendishServant])
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_2.store, [Scallywag, Scallywag, Scallywag])
        self.assertCardListEquals(player_2.hand, [])
        self.assertCardListEquals(player_2.in_play, [])
        for player_name, player in tavern.players.items():
            self.assertEqual(player.coins, 3, f"{player.name} does not have the right number of coins")
        player_1.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.store, [FiendishServant, FiendishServant])
        self.assertCardListEquals(player_1.hand, [FiendishServant])
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_2.store, [Scallywag, Scallywag])
        self.assertCardListEquals(player_2.hand, [Scallywag])
        self.assertCardListEquals(player_2.in_play, [])
        for player_name, player in tavern.players.items():
            self.assertEqual(player.coins, 0, f"{player.name} does not have the right number of coins")
        player_1.summon_from_hand(HandIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.store, [FiendishServant, FiendishServant])
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [FiendishServant])
        self.assertCardListEquals(player_2.store, [Scallywag, Scallywag])
        self.assertCardListEquals(player_2.hand, [])
        self.assertCardListEquals(player_2.in_play, [Scallywag])
        tavern.combat_step()
        self.assertEqual(player_1.health, 38, f"{player_1.name}'s heath is incorrect")
        self.assertEqual(player_2.health, 40, f"{player_2.name}'s heath is incorrect")

    class TestTwoRoundsRandomizer(DefaultRandomizer):
        def __init__(self):
            super().__init__()
            self.cards_drawn = 0

        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number == 0:
                if player_name == "Dante_Kong":
                    return force_card(cards, FiendishServant)

                if player_name == "lucy":
                    return force_card(cards, Scallywag)
            elif round_number == 1:
                if player_name == "Dante_Kong":
                    return force_card(cards, FiendishServant)

                if player_name == "lucy":
                    return force_card(cards, FiendishServant)

        def select_attack_target(self, defenders: List[MonsterCard]) -> MonsterCard:
            target = [card for card in defenders if type(card) is not Scallywag]
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
        self.assertCardListEquals(player_1.store, [FiendishServant, FiendishServant, FiendishServant])
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_2.store, [Scallywag, Scallywag, Scallywag])
        self.assertCardListEquals(player_2.hand, [])
        self.assertCardListEquals(player_2.in_play, [])
        for player_name, player in tavern.players.items():
            self.assertEqual(player.coins, 3, f"{player.name} does not have the right number of coins")
        player_1.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.store, [FiendishServant, FiendishServant])
        self.assertCardListEquals(player_1.hand, [FiendishServant])
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_2.store, [Scallywag, Scallywag])
        self.assertCardListEquals(player_2.hand, [Scallywag])
        self.assertCardListEquals(player_2.in_play, [])
        for player_name, player in tavern.players.items():
            self.assertEqual(player.coins, 0, f"{player.name} does not have the right number of coins")
        player_1.summon_from_hand(HandIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.store, [FiendishServant, FiendishServant])
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [FiendishServant])
        self.assertCardListEquals(player_2.store, [Scallywag, Scallywag])
        self.assertCardListEquals(player_2.hand, [])
        self.assertCardListEquals(player_2.in_play, [Scallywag])
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

    class TestWrathWeaverRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
            if round_number == 0:
                return [card for card in cards if type(card) is WrathWeaver][0]
            else:
                return [card for card in cards if type(card) is FiendishServant][0]

    def test_wrath_weaver(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestWrathWeaverRandomizer()
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
        tavern.randomizer = CardForcer([FiendishServant, WrathWeaver, WrathWeaver] * 4)
        player_1 = tavern.add_player_with_hero("Dante_Kong", LordJaraxxus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.combat_step()

        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0))

        self.assertCardListEquals(player_1.in_play, [FiendishServant, WrathWeaver])
        player_1.hero_power()
        fiendish_servant_stats = (player_1.in_play[0].attack, player_1.in_play[0].health)
        wrath_weaver_stats = (player_1.in_play[1].attack, player_1.in_play[1].health)
        self.assertEqual(fiendish_servant_stats, (3, 2))
        self.assertEqual(wrath_weaver_stats, (1, 3))

    def test_patchwerk(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", PatchWerk())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(player_1.health, 55)
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
        self.assertEqual(len(player_1.spells), 0)
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.hand, [])
        self.assertCardListEquals(player_1.in_play, [TabbyCat, AlleyCat, TabbyCat])
        self.assertEqual(player_1.in_play[1].golden, True)
        self.assertEqual(player_1.in_play[2].golden, True)
        self.assertEqual(player_1.in_play[1].attack, 2)
        self.assertEqual(player_1.in_play[2].attack, 2)
        self.assertEqual(player_1.in_play[1].health, 2)
        self.assertEqual(player_1.in_play[2].health, 2)
        self.assertCardListEquals(player_1.spells, [TripleRewardCard])
        self.assertEqual(player_1.spells[0].tier, 2)

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
        self.assertEqual(len(player_1.spells), 0)
        player_1.summon_from_hand(HandIndex(0))

        self.assertCardListEquals(player_1.hand, [TabbyCat])
        self.assertEqual(player_1.hand[0].golden, True)
        self.assertCardListEquals(player_1.in_play, [AlleyCat])
        self.assertEqual(player_1.in_play[0].golden, False)
        self.assertEqual(len(player_1.spells), 0)
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.spells, [TripleRewardCard])
        self.assertEqual(player_1.spells[0].tier, 2)

    def test_buffed_golden(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = CardForcer([DeckSwabbie] * 18)
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
        self.assertCardListEquals(player_1.hand, [DeckSwabbie])
        self.assertEqual(player_1.hand[0].golden, True)
        self.assertEqual(player_1.hand[0].attack, 4)
        self.assertEqual(player_1.hand[0].health, 8)
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
        self.assertCardListEquals(player_1.spells, [TripleRewardCard])
        self.assertEqual(player_1.spells[0].tier, 2)
        player_1.play_spell(SpellIndex(0))
        player_1.select_discover(DiscoverIndex(0))
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
        self.assertCardListEquals(player_1.spells, [TripleRewardCard])
        self.assertEqual(player_1.spells[0].tier, 2)
        player_1.upgrade_tavern()
        player_2.upgrade_tavern()
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.play_spell(SpellIndex(0))
        player_1.select_discover(DiscoverIndex(0))
        print(player_1.hand)
        self.assertCardListEquals(player_1.hand, [FreedealingGambler])
        self.assertEqual(player_1.hand[0].golden, True)
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.spells, [TripleRewardCard])
        self.assertEqual(player_1.spells[0].tier, 3)

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
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = RepeatedCardForcer([KaboomBot, MetaltoothLeaper])
        for _ in range(2):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [KaboomBot, KaboomBot])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [KaboomBot, KaboomBot, MetaltoothLeaper])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health)

    def test_rabid_saurolisk(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = CardForcer([Scallywag, AlleyCat, AlleyCat] * 6 + [RabidSaurolisk] * 8)
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
        self.assertCardListEquals(player_1.in_play, [RabidSaurolisk, AlleyCat, TabbyCat, Scallywag])
        self.assertEqual(player_1.in_play[0].attack, 4)
        self.assertEqual(player_1.in_play[0].health, 4)

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
        self.assertEqual(player_1.store[0].attack, 5)
        self.assertEqual(player_1.store[0].health, 4)

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
        tavern.randomizer = CardForcer(
            [Scallywag, MurlocTidehunter, MicroMachine, FiendishServant, FiendishServant, DeckSwabbie] * 2)
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
        tavern.randomizer = CardForcer([MurlocTidehunter, RockpoolHunter, RockpoolHunter] * 4)
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
        tavern.randomizer = CardForcer([MicroMummy] * 6)
        tavern.buying_step()
        for card in player_1.store:
            self.assertEqual(card.attack, card.base_attack + 1)
            self.assertEqual(card.health, card.base_health + 1)
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
        self.assertEqual(player_1.coins, 2)
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
        tavern.randomizer = CardForcer([FiendishServant] * (18 + 16))
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = CardForcer([CrystalWeaver] * 10)
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

    def test_bloodsail_cannoneer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        tavern.randomizer = RepeatedCardForcer([Scallywag])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([WrathWeaver])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([BloodsailCannoneer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Scallywag, WrathWeaver, BloodsailCannoneer])
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
        tavern.randomizer = CardForcer([KaboomBot] * 16)
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
        self.assertEqual(player_1.in_play[0].attack, 2)
        self.assertEqual(player_1.in_play[0].health, 2)
        self.assertEqual(player_1.in_play[1].attack, 2)
        self.assertEqual(player_1.in_play[1].health, 2)
        self.assertCardListEquals(player_1.in_play, [KaboomBot, KaboomBot])
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [KaboomBot, KaboomBot, ScrewjankClunker])
        self.assertEqual(player_1.in_play[0].attack, 4)
        self.assertEqual(player_1.in_play[0].health, 4)
        self.assertEqual(player_1.in_play[1].attack, 2)
        self.assertEqual(player_1.in_play[1].health, 2)
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
        self.assertEqual(player_1.in_play[0].attack, 3)
        self.assertEqual(player_1.in_play[0].health, 4)
        self.assertEqual(player_1.in_play[1].attack, 5)
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
        self.assertCardListEquals(player_1.in_play, [Khadgar, Khadgar, AlleyCat, TabbyCat])
        self.assertCardListEquals(player_1.hand, [TabbyCat])
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
        self.assertEqual(player_1.health, 38)
        self.assertCardListEquals(player_1.spells, [GoldCoin])
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(player_1.coins, 4)
        self.assertEqual(len(player_1.spells), 0)

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
                return force_card(cards, MicroMummy)
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
        self.assertCardListEquals(player_1.hand,
                                  [ScavengingHyena, MicroMummy, Scallywag, DragonspawnLieutenant, FiendishServant,
                                   MurlocTidecaller])
        for i in range(player_1.hand_size()):
            self.assertEqual(player_1.hand[i].attack, player_1.hand[i].base_attack + 2)
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
        player_1.reroll_store()
        self.assertEqual(len(player_1.store), 4)
        self.assertEqual(type(player_1.store[3]), DragonspawnLieutenant)

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
        tavern.randomizer = RepeatedCardForcer([Scallywag])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = CardForcer([StrongshellScavenger, AlleyCat] * 5)
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant, Scallywag, StrongshellScavenger])
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
        self.assertCardListEquals(player_1.in_play,
                                  [MurlocTidehunter, MurlocScout, DragonspawnLieutenant, KingBagurgle])
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
        self.assertCardListEquals(player_1.in_play,
                                  [MurlocTidehunter, MurlocScout, DragonspawnLieutenant, RazorgoreTheUntamed])
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
        tavern.randomizer = RepeatedCardForcer([Scallywag, RockpoolHunter])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play,
                                  [DragonspawnLieutenant, KalecgosArcaneAspect, Scallywag, RockpoolHunter])
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
        tavern = Tavern(restrict_types=False, include_graveyard=True)
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
        self.assertEqual(player_1.in_play[0].health, 5)
        self.assertEqual(player_1.in_play[0].attack, 5)
        self.assertEqual(player_1.in_play[1].health, 6)
        self.assertEqual(player_1.in_play[1].attack, 6)
        self.assertEqual(player_1.in_play[2].health, 6)
        self.assertEqual(player_1.in_play[2].attack, 6)

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
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([AlleyCat, KaboomBot])
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
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, KaboomBot, IronSensei])
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
                return force_card(cards, KaboomBot)
            else:
                return cards[0]

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
                                                     ScavengingHyena, KaboomBot, LightfangEnforcer])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 2)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 2)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 2)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 2)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health + 2)
        self.assertEqual(player_1.in_play[4].attack, player_1.in_play[4].base_attack + 2)
        self.assertEqual(player_1.in_play[4].health, player_1.in_play[4].base_health + 2)
        self.assertEqual(player_1.in_play[5].attack, player_1.in_play[5].base_attack + 2)
        self.assertEqual(player_1.in_play[5].health, player_1.in_play[5].base_health + 2)
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
                return force_card(cards, Sellemental)
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
                                                     DragonspawnLieutenant, Sellemental, MenagerieMug])
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
                                                     DragonspawnLieutenant, Sellemental, MenagerieJug])
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
                                                     DragonspawnLieutenant, Sellemental, AlleyCat, TabbyCat])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 1)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 1)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 1)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health + 1)
        self.assertEqual(player_1.in_play[4].attack, player_1.in_play[4].base_attack + 1)
        self.assertEqual(player_1.in_play[4].health, player_1.in_play[4].base_health + 1)
        self.assertEqual(player_1.in_play[5].attack, player_1.in_play[5].base_attack + 1)
        self.assertEqual(player_1.in_play[5].health, player_1.in_play[5].base_health + 1)
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
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        tavern.randomizer = self.TestAmalgadonRandomizer()
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Amalgam, AlleyCat, TabbyCat, Amalgadon])
        self.assertTrue(player_1.in_play[3].divine_shield)
        self.assertEqual(len(player_1.in_play[3].deathrattles), 1)

    def test_only_amalgadon(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([Amalgadon, AlleyCat, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        tavern.randomizer = self.TestAmalgadonRandomizer()
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Amalgadon])
        self.assertFalse(player_1.in_play[0].divine_shield)

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
        tavern.randomizer = RepeatedCardForcer([MicroMummy, AnnoyOModule, AnnoyOModule])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [MicroMummy])
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
        tavern.randomizer = RepeatedCardForcer([MicroMummy])
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
        self.assertCardListEquals(player_1.in_play, [MicroMummy])
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
        tavern.randomizer = RepeatedCardForcer([MicroMachine, KalecgosArcaneAspect])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_2.purchase(StoreIndex(1))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([ReplicatingMenace])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        self.assertCardListEquals(player_1.hand, [MicroMachine, ReplicatingMenace])
        self.assertCardListEquals(player_2.hand, [KalecgosArcaneAspect])
        self.assertTrue(player_1.hand[1].golden)
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [MicroMachine])
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
        tavern.randomizer = RepeatedCardForcer([MicroMummy, ReplicatingMenace])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [MicroMummy])
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
        tavern.randomizer = RepeatedCardForcer([MicroMummy, ReplicatingMenace])
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
        self.assertEqual(len(player_1.in_play), 0)
        self.assertEqual(len(player_1.discover_queue[0]), 2)
        for card in player_1.discover_queue[0]:
            self.assertEqual(card.tier, 2)

    def test_hooktusk_tier_one_minion(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", CaptainHooktusk())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie])
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertEqual(len(player_1.in_play), 0)
        self.assertEqual(len(player_1.discover_queue[0]), 2)
        for card in player_1.discover_queue[0]:
            self.assertEqual(card.tier, 1)

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
        self.assertEqual(len(player_1.spells), 0)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.coins, 0)
        self.assertEqual(len(player_1.spells), 0)
        tavern.combat_step()
        tavern.buying_step()
        self.assertCardListEquals(player_1.spells, [TripleRewardCard, TripleRewardCard])
        for reward in player_1.spells:
            self.assertEqual(reward.tier, 3)

    def test_southsea_strongarm(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
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
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 4)
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
        self.assertCardListEquals(player_1.spells, [TripleRewardCard])
        for _ in range(2):
            player_1.select_discover(DiscoverIndex(0))
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(len(player_1.discover_queue), 1)
        player_1.select_discover(DiscoverIndex(0))
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
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant, Murozond, Goldgrubber])
        self.assertTrue(player_1.in_play[1].golden)
        self.assertTrue(player_1.in_play[2].golden)
        self.assertEqual(len(player_1.spells), 2)

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

    def test_dinotamer_brann(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", DinotamerBrann())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        for i in range(5):
            tavern.buying_step()
            self.assertTrue(BrannBronzebeard not in [type(card) for card in player_1.hand])
            player_1.purchase(StoreIndex(0))
            tavern.combat_step()
        self.assertCardListEquals(player_1.hand, [AlleyCat, AlleyCat, AlleyCat, BrannBronzebeard])

    def test_alexstrasza(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Alexstrasza())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        self.assertEqual(len(player_1.discover_queue), 2)
        for _ in range(2):
            self.assertTrue(card.check_type(MONSTER_TYPES.DRAGON) for card in player_1.discover_queue[0])
            player_1.select_discover(DiscoverIndex(0))
        self.assertEqual(len(player_1.hand), 2)
        self.assertEqual(len(player_1.discover_queue), 0)

    class TestKingMuklaRandomizer(DefaultRandomizer):
        def select_random_number(self, lo: int, hi: int) -> int:
            return 2

    def test_king_mukla(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestKingMuklaRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", KingMukla())
        player_2 = tavern.add_player_with_hero("lucy")
        player_3 = tavern.add_player_with_hero("thing_1")
        player_4 = tavern.add_player_with_hero("thing_2")
        tavern.buying_step()
        player_1.hero_power()
        self.assertCardListEquals(player_1.spells, [Banana, Banana])
        for player in list(tavern.players.values())[1:]:
            self.assertEqual(len(player.spells), 0)
        tavern.combat_step()
        self.assertCardListEquals(player_1.spells, [Banana, Banana])
        for player in list(tavern.players.values())[1:]:
            self.assertCardListEquals(player.spells, [Banana])

        tavern.buying_step()
        player_1.play_spell(SpellIndex(0), store_index=StoreIndex(0))
        player_1.play_spell(SpellIndex(0), store_index=StoreIndex(0))
        self.assertEqual(len(player_1.spells), 0)
        self.assertEqual(player_1.store[0].attack, player_1.store[0].base_attack + 2)
        self.assertEqual(player_1.store[0].health, player_1.store[0].base_health + 2)

    def test_elise_starseeker(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", EliseStarseeker())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 2)
        self.assertCardListEquals(player_1.spells, [RecruitmentMap])
        self.assertEqual(player_1.spells[0].tier, 2)
        tavern.buying_step()
        self.assertEqual(player_1.coins, 5)
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(player_1.coins, 2)
        self.assertEqual(len(player_1.discover_queue), 1)
        self.assertTrue(all(card.tier == 2 for card in player_1.discover_queue[0]))

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
        player_1.hero_power(BoardIndex(0))
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
        self.assertEqual(len(player_2.in_play), 3)
        self.assertEqual(len(player_1.discover_queue), 1)
        self.assertEqual(len(player_1.discover_queue[0]), 3)
        for card in player_1.discover_queue[0]:
            self.assertTrue(card.token)
            if type(card) != DefenderOfArgus:
                self.assertEqual(card.attack, card.base_attack + 1)
                self.assertEqual(card.health, card.base_health + 1)
                self.assertTrue(card.taunt)
        player_1.select_discover(DiscoverIndex(0))
        self.assertEqual(len(player_1.hand), 1)
        self.assertFalse(player_1.hand[0].token)
        self.assertEqual(len(player_1.discover_queue), 0)
        tracked_card = player_1.hand[0]
        in_pool = len([card for card in tavern.deck.all_cards() if type(card) == type(tracked_card)])
        player_1.sell_minion(BoardIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == type(tracked_card)]),
                         in_pool + 1)

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
        self.assertTrue(
            player_1.hand[0].check_type(MONSTER_TYPES.ELEMENTAL) and player_1.hand[0].tier <= player_1.tavern_tier)

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
        tavern.combat_step()
        tavern.buying_step()
        for card in player_1.store:
            if card.check_type(MONSTER_TYPES.ELEMENTAL):
                self.assertEqual(card.attack, card.base_attack + 2)
                self.assertEqual(card.health, card.base_health + 2)

    def test_nomi_jandice(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", JandiceBarov())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer(
            [NomiKitchenNightmare, Sellemental, Sellemental, Sellemental, Sellemental])
        tavern.buying_step()
        for _ in range(2):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.nomi_bonus, 1)
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
        self.assertEqual(player_1.nomi_bonus, 2)
        self.assertCardListEquals(player_1.in_play, [NomiKitchenNightmare, Sellemental, Sellemental])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 1)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 1)
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([RefreshingAnomaly])
        tavern.buying_step()
        player_1.hero_power(BoardIndex(2))
        player_1.purchase(StoreIndex(4))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [NomiKitchenNightmare, Sellemental, RefreshingAnomaly, Sellemental])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 2)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 2)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack + 2)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health + 2)

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
        player_1.hero_power(store_index=StoreIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        self.assertFalse(player_1.store[0].frozen)
        self.assertEqual(player_1.store[0].attack, player_1.store[0].base_attack + 2)
        self.assertEqual(player_1.store[0].health, player_1.store[0].base_health + 1)
        self.assertEqual(player_1.store[1].attack, player_1.store[1].base_attack)
        self.assertEqual(player_1.store[1].health, player_1.store[1].base_health)
        self.assertEqual(player_1.store[2].attack, player_1.store[2].base_attack)
        self.assertEqual(player_1.store[2].health, player_1.store[2].base_health)

    def test_galakrond(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Galakrond())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertEqual(player_1.store[2].tier, 2)
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([Amalgadon, AlleyCat])
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
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
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([StasisElemental])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(len(player_1.store), 4)
        self.assertFalse(player_1.store[0].frozen)
        self.assertFalse(player_1.store[1].frozen)
        self.assertFalse(player_1.store[2].frozen)
        self.assertTrue(player_1.store[3].frozen)
        player_1.reroll_store()
        for card in player_1.store:
            self.assertEqual(type(card), StasisElemental)
            self.assertFalse(card.frozen)

    def test_the_lich_king(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheLichKing())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([FiendishServant, AlleyCat, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand((HandIndex(0)))
        player_2.purchase(StoreIndex(1))
        player_2.summon_from_hand((HandIndex(0)))
        self.assertCardListEquals(player_1.in_play, [FiendishServant])
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
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        tavern.randomizer = self.TestGoldenAmalgadonRandomizer()
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, Amalgadon])
        self.assertTrue(player_1.in_play[2].windfury)
        self.assertFalse(player_1.in_play[2].mega_windfury)

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
        self.assertCardListEquals(player_2.in_play,
                                  [AlleyCat, TabbyCat, MurlocTidehunter, MurlocScout, DragonspawnLieutenant])
        tavern.buying_step()
        player_1.hero_power()
        self.assertCardListEquals(player_1.store,
                                  [AlleyCat, TabbyCat, MurlocTidehunter, MurlocScout, DragonspawnLieutenant])

    def test_tess_extra_cards(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TessGreymane())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant])
        tavern.buying_step()
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_2.in_play, [DragonspawnLieutenant])
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(player_1.store_size(), 3)
        self.assertEqual(type(player_1.store[0]), DragonspawnLieutenant)

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

    def test_zephrys_the_great(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", ZephrysTheGreat())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([Sellemental])
        for _ in range(2):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        tavern.buying_step()
        self.assertCardListEquals(player_1.in_play, [Sellemental, Sellemental])
        player_1.hero_power()
        self.assertEqual(player_1.hero.wishes_left, 2)
        self.assertCardListEquals(player_1.in_play, [])
        self.assertCardListEquals(player_1.hand, [Sellemental])
        self.assertTrue(player_1.hand[0].golden)

    class TestSilasDarkmoonRandomizer(DefaultRandomizer):
        def select_random_number(self, lo, hi) -> int:
            return 1

    def test_silas_darkmoon(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", SilasDarkmoon())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = self.TestSilasDarkmoonRandomizer()
        for i in range(3):
            tavern.buying_step()
            self.assertEqual(player_1.hero.tickets_purchased, i)
            player_1.purchase(StoreIndex(0))
            tavern.combat_step()
        self.assertEqual(player_1.hero.tickets_purchased, 0)
        self.assertCardListEquals(player_1.spells, [Prize])
        self.assertEqual(player_1.spells[0].tier, 1)
        player_1.play_spell(SpellIndex(0))
        self.assertTrue(all(card.tier == 1 for card in player_1.discover_queue[0]))

    class TestBigBananaRandomizer(DefaultRandomizer):
        def select_random_number(self, lo: int, hi: int) -> int:
            return 1

    def test_big_banana(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestBigBananaRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", KingMukla())
        player_2 = tavern.add_player_with_hero("lucy")
        player_3 = tavern.add_player_with_hero("thing_1")
        player_4 = tavern.add_player_with_hero("thing_2")
        tavern.buying_step()
        player_1.hero_power()
        self.assertCardListEquals(player_1.spells, [BigBanana, BigBanana])
        for player in list(tavern.players.values())[1:]:
            self.assertEqual(len(player.spells), 0)
        tavern.combat_step()
        self.assertCardListEquals(player_1.spells, [BigBanana, BigBanana])
        for player in list(tavern.players.values())[1:]:
            self.assertCardListEquals(player.spells, [Banana])

        tavern.buying_step()
        player_1.play_spell(SpellIndex(0), store_index=StoreIndex(0))
        player_1.play_spell(SpellIndex(0), store_index=StoreIndex(0))
        self.assertEqual(len(player_1.spells), 0)
        self.assertEqual(player_1.store[0].attack, player_1.store[0].base_attack + 4)
        self.assertEqual(player_1.store[0].health, player_1.store[0].base_health + 4)

    def test_buy_phase_death(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", LichBazhial())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.health = 2
        player_1.hero_power()
        self.assertEqual(player_1.health, 0)
        self.assertTrue(player_1.dead)
        for action in generate_standard_actions(player_1):
            if type(action) != EndPhaseAction:
                self.assertFalse(action.valid(player_1))

    def test_the_great_akazamzarak(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheGreatAkazamzarak())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(len(player_1.hero.discover_queue[0]), 3)
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(len(player_1.secrets), 1)
        self.assertEqual(len(player_1.hero.discover_queue), 0)

    class TestAkazamzarakIceBlockRandomizer(DefaultRandomizer):
        def select_secret(self, secrets: List[Type['Secret']]) -> Type['Secret']:
            if BaseSecret.IceBlock in secrets:
                return BaseSecret.IceBlock
            else:
                return secrets[0]

    def test_ice_block(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestAkazamzarakIceBlockRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheGreatAkazamzarak())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power()
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.secrets[0]), BaseSecret.IceBlock)
        player_1.health = 1
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertFalse(player_1.hero.give_immunity)
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([VulgarHomunculus])
        tavern.buying_step()
        self.assertFalse(player_1.dead)
        self.assertEqual(player_1.health, 1)
        self.assertEqual(len(player_1.secrets), 0)
        self.assertTrue(player_1.hero.give_immunity)
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.health, 1)
        tavern.combat_step()
        self.assertFalse(player_1.hero.give_immunity)

    def test_ice_block_available_once(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestAkazamzarakIceBlockRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheGreatAkazamzarak())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertFalse(player_1.hero.discovered_ice_block)
        player_1.hero_power()
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.secrets[0]), BaseSecret.IceBlock)
        self.assertTrue(player_1.hero.discovered_ice_block)
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.secrets[0]), BaseSecret.IceBlock)
        self.assertNotEqual(player_1.secrets[1], BaseSecret.IceBlock)

    class TestAkazamzarakCompetetiveSpiritRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
            return force_card(cards, AlleyCat)

        def select_secret(self, secrets: List[Type['Secret']]) -> Type['Secret']:
            if BaseSecret.CompetitiveSpirit in secrets:
                return BaseSecret.CompetitiveSpirit
            else:
                return secrets[0]

    def test_competetive_spirit(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestAkazamzarakCompetetiveSpiritRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", TheGreatAkazamzarak())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.secrets[0]), BaseSecret.CompetitiveSpirit)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(len(player_1.secrets), 0)
        for card in player_1.in_play:
            self.assertEqual(card.attack, card.base_attack + 1)
            self.assertEqual(card.health, card.base_health + 1)

    def test_sir_finley_mrrgglton(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", SirFinleyMrrgglton())
        player_2 = tavern.add_player_with_hero("lucy")
        for hero in tavern.hero_pool:
            self.assertNotEqual(type(hero), SirFinleyMrrgglton)
        tavern.buying_step()
        self.assertEqual(len(player_1.hero.discover_queue[0]), 3)
        self.assertEqual(type(player_1.hero), SirFinleyMrrgglton)
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertNotEqual(type(player_1.hero), SirFinleyMrrgglton)

    # class TestLordBarovRandomizer(DefaultRandomizer):
    #     def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
    #         if player_name == "Dante_Kong":
    #             return force_card(cards, RockpoolHunter)
    #         if player_name == "lucy":
    #             return force_card(cards, DeckSwabbie)
    #         return force_card(cards, AlleyCat)
    #
    #     def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
    #         return [(players[0], players[1])]
    #
    # def test_lord_barov(self):
    #     tavern = Tavern(restrict_types=False)
    #     player_1 = tavern.add_player_with_hero("Dante_Kong", LordBarov())
    #     player_2 = tavern.add_player_with_hero("lucy")
    #     tavern.randomizer = self.TestLordBarovRandomizer()
    #     tavern.buying_step()
    #     tavern.combat_step()
    #     tavern.buying_step()
    #     player_1.purchase(StoreIndex(0))
    #     player_1.summon_from_hand(HandIndex(0))
    #     player_2.purchase(StoreIndex(1))
    #     player_2.summon_from_hand(HandIndex(0))
    #     self.assertCardListEquals(player_1.in_play, [RockpoolHunter])
    #     self.assertCardListEquals(player_2.in_play, [DeckSwabbie])
    #     player_1.hero_power()
    #     self.assertEqual(len(player_1.hero.discover_queue[0]), 2)
    #     self.assertEqual(tavern.current_player_pairings, [(player_1, player_2)])
    #     player_1.hero_select_discover(DiscoverIndex(0))
    #     self.assertEqual(player_1.hero.winning_pick, player_1)
    #     tavern.combat_step()
    #     self.assertCardListEquals(player_1.spells, [GoldCoin, GoldCoin, GoldCoin])
    #     self.assertIsNone(player_1.hero.winning_pick)
    #
    # def test_lord_barov_loser(self):
    #     tavern = Tavern(restrict_types=False)
    #     player_1 = tavern.add_player_with_hero("Dante_Kong", LordBarov())
    #     player_2 = tavern.add_player_with_hero("lucy")
    #     tavern.randomizer = self.TestLordBarovRandomizer()
    #     tavern.buying_step()
    #     tavern.combat_step()
    #     tavern.buying_step()
    #     player_1.purchase(StoreIndex(0))
    #     player_1.summon_from_hand(HandIndex(0))
    #     player_2.purchase(StoreIndex(1))
    #     player_2.summon_from_hand(HandIndex(0))
    #     self.assertCardListEquals(player_1.in_play, [RockpoolHunter])
    #     self.assertCardListEquals(player_2.in_play, [DeckSwabbie])
    #     player_1.hero_power()
    #     self.assertEqual(len(player_1.hero.discover_queue[0]), 2)
    #     self.assertEqual(tavern.current_player_pairings, [(player_1, player_2)])
    #     player_1.hero_select_discover(DiscoverIndex(1))
    #     self.assertEqual(player_1.hero.winning_pick, player_2)
    #     tavern.combat_step()
    #     self.assertEqual(len(player_1.spells), 0)
    #     self.assertIsNone(player_1.hero.winning_pick)
    #
    # def test_lord_barov_4_players(self):
    #     tavern = Tavern(restrict_types=False)
    #     player_1 = tavern.add_player_with_hero("Dante_Kong")
    #     player_2 = tavern.add_player_with_hero("lucy")
    #     player_3 = tavern.add_player_with_hero("player3", LordBarov())
    #     player_4 = tavern.add_player_with_hero("player4")
    #     tavern.randomizer = self.TestLordBarovRandomizer()
    #     tavern.buying_step()
    #     player_1.purchase(StoreIndex(0))
    #     player_1.summon_from_hand(HandIndex(0))
    #     player_2.purchase(StoreIndex(1))
    #     player_2.summon_from_hand(HandIndex(0))
    #     self.assertCardListEquals(player_1.in_play, [RockpoolHunter])
    #     self.assertCardListEquals(player_2.in_play, [DeckSwabbie])
    #     player_3.hero_power()
    #     self.assertEqual(len(player_3.hero.discover_queue[0]), 2)
    #     self.assertEqual(tavern.current_player_pairings, [(player_1, player_2)])
    #     player_3.hero_select_discover(DiscoverIndex(0))
    #     self.assertEqual(player_3.hero.winning_pick, player_1)
    #     tavern.combat_step()
    #     self.assertCardListEquals(player_3.spells, [GoldCoin, GoldCoin, GoldCoin])
    #     self.assertIsNone(player_3.hero.winning_pick)
    #
    # class TestLordBarovTieRandomizer(DefaultRandomizer):
    #     def select_draw_card(self, cards: List[MonsterCard], player_name: str, round_number: int) -> MonsterCard:
    #         return force_card(cards, DeckSwabbie)
    #
    #     def select_player_pairings(self, players: List['Player']) -> List[Tuple['Player', 'Player']]:
    #         return [(players[0], players[1])]
    #
    # def test_lord_barov_tie(self):
    #     tavern = Tavern(restrict_types=False)
    #     player_1 = tavern.add_player_with_hero("Dante_Kong", LordBarov())
    #     player_2 = tavern.add_player_with_hero("lucy")
    #     tavern.randomizer = self.TestLordBarovTieRandomizer()
    #     tavern.buying_step()
    #     tavern.combat_step()
    #     tavern.buying_step()
    #     player_1.purchase(StoreIndex(0))
    #     player_1.summon_from_hand(HandIndex(0))
    #     player_2.purchase(StoreIndex(1))
    #     player_2.summon_from_hand(HandIndex(0))
    #     self.assertCardListEquals(player_1.in_play, [DeckSwabbie])
    #     self.assertCardListEquals(player_2.in_play, [DeckSwabbie])
    #     player_1.hero_power()
    #     self.assertEqual(len(player_1.hero.discover_queue[0]), 2)
    #     self.assertEqual(tavern.current_player_pairings, [(player_1, player_2)])
    #     player_1.hero_select_discover(DiscoverIndex(0))
    #     self.assertEqual(player_1.hero.winning_pick, player_1)
    #     tavern.combat_step()
    #     self.assertCardListEquals(player_1.spells, [GoldCoin])
    #     self.assertIsNone(player_1.hero.winning_pick)

    class TestMrrggltonRatKingRandomizer(DefaultRandomizer):
        def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
            if TheRatKing in [type(hero) for hero in hero_pool]:
                return [hero for hero in hero_pool if isinstance(hero, TheRatKing)][0]
            else:
                return hero_pool[0]

    def test_mrrgglton_choose_rat_king(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestMrrggltonRatKingRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", SirFinleyMrrgglton())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(type(player_1.hero), SirFinleyMrrgglton)
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.hero), TheRatKing)
        self.assertIsNotNone(player_1.hero.current_type)

    class TestMrrggltonMillhouseRandomizer(DefaultRandomizer):
        def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
            if MillhouseManastorm in [type(hero) for hero in hero_pool]:
                return [hero for hero in hero_pool if isinstance(hero, MillhouseManastorm)][0]
            else:
                return hero_pool[0]

    def test_mrrgglton_choose_millhouse(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestMrrggltonMillhouseRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", SirFinleyMrrgglton())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(type(player_1.hero), SirFinleyMrrgglton)
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.hero), MillhouseManastorm)
        self.assertEqual(player_1.minion_cost, 2)
        self.assertEqual(player_1.refresh_store_cost, 2)
        self.assertEqual(player_1.tavern_upgrade_cost, 6)

    class TestMrrggltonCuratorRandomizer(DefaultRandomizer):
        def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
            if TheCurator in [type(hero) for hero in hero_pool]:
                return [hero for hero in hero_pool if isinstance(hero, TheCurator)][0]
            else:
                return hero_pool[0]

    def test_mrrgglton_choose_curator(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestMrrggltonCuratorRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", SirFinleyMrrgglton())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertEqual(type(player_1.hero), SirFinleyMrrgglton)
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.hero), TheCurator)
        self.assertCardListEquals(player_1.in_play, [Amalgam])

    def test_maiev_shadowsong(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", MaievShadowsong())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertEqual(len(player_1.hero.dormant_minions), 1)
        player_1.reroll_store()
        for _ in range(3):
            self.assertEqual(len(player_1.store), 3)
            self.assertEqual(len(player_1.hero.dormant_minions), 1)
            tavern.combat_step()
            tavern.buying_step()
        self.assertEqual(len(player_1.store), 3)
        self.assertEqual(player_1.hand_size(), 1)
        self.assertEqual(len(player_1.hero.dormant_minions), 0)
        self.assertEqual(player_1.hand[0].attack, player_1.hand[0].base_attack + 1)
        self.assertEqual(player_1.hand[0].health, player_1.hand[0].base_health + 1)
        tavern.combat_step()
        self.upgrade_to_tier(tavern, 6)
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertEqual(len(player_1.hero.dormant_minions), 1)
        player_1.reroll_store()
        self.assertEqual(len(player_1.store), 6)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(len(player_1.store), 6)
        self.assertEqual(len(player_1.hero.dormant_minions), 1)
        player_1.hero_power(store_index=StoreIndex(1))
        self.assertEqual(len(player_1.hero.dormant_minions), 2)
        player_1.reroll_store()
        self.assertEqual(len(player_1.store), 5)
        self.assertEqual(len(player_1.hero.dormant_minions), 2)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(len(player_1.hero.dormant_minions), 2)
        self.assertEqual(player_1.hand_size(), 1)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(len(player_1.hero.dormant_minions), 1)
        self.assertEqual(player_1.hand_size(), 2)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(len(player_1.hero.dormant_minions), 0)
        self.assertEqual(player_1.hand_size(), 3)

    def test_reno_adds_one_copy(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", RenoJackson())
        player_2 = tavern.add_player_with_hero("lucy")
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == WrathWeaver]), 16)
        tavern.randomizer = RepeatedCardForcer([WrathWeaver, AlleyCat, AlleyCat])
        tavern.buying_step()
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == WrathWeaver]), 14)
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertTrue(player_1.in_play[0].golden)
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == WrathWeaver]), 15)
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == WrathWeaver]), 16)

    def test_tess_minion_pool_interaction(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", TessGreymane())
        player_2 = tavern.add_player_with_hero("lucy")
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == AlleyCat]), 16)
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie])
        tavern.buying_step()
        player_1.hero_power()
        self.assertEqual(player_1.store_size(), 3)
        self.assertCardListEquals(player_1.store[:2], [AlleyCat, TabbyCat])
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == AlleyCat]), 15)
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertFalse(player_1.in_play[0].token)
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == AlleyCat]), 15)
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == TabbyCat]), 0)
        player_1.hero_power()
        self.assertEqual(player_1.store_size(), 3)
        self.assertCardListEquals(player_1.store[:2], [AlleyCat, TabbyCat])
        player_2.sell_minion(BoardIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == AlleyCat]), 17)
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [TabbyCat, TabbyCat])
        self.assertTrue(player_1.in_play[0].token)
        self.assertTrue(player_1.in_play[1].token)
        player_1.sell_minion(BoardIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == TabbyCat]), 0)

    def test_rafaam_pool_interaction(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", ArchVillianRafaam())
        player_2 = tavern.add_player_with_hero("lucy")
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == DeckSwabbie]), 16)
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        player_1.sell_minion(BoardIndex(0))
        player_2.sell_minion(BoardIndex(0))
        self.assertEqual(len([card for card in tavern.deck.all_cards() if type(card) == DeckSwabbie]), 17)

    def test_cthun(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", CThun())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DeckSwabbie])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health)
        tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        tavern.buying_step()
        player_1.hero_power()
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 3)

    class TestYShaarjRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
            return force_card(cards, DeckSwabbie)

        def select_gain_card(self, cards: List['MonsterCard']) -> 'MonsterCard':
            minion_types = [type(card) for card in cards]
            if MurlocTidehunter in minion_types:
                return force_card(cards, MurlocTidehunter)
            else:
                return cards[0]

    def test_yshaarj(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", YShaarj())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = self.TestYShaarjRandomizer()
        tavern.buying_step()
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        player_1.hero_power()
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 40)
        self.assertCardListEquals(player_1.hand, [MurlocTidehunter])

    class TestYShaarjCombatSummonRandomizer(DefaultRandomizer):
        def select_draw_card(self, cards: List['MonsterCard'], player_name: str, round_number: int) -> 'MonsterCard':
            minion_types = [type(card) for card in cards]
            if player_name == "lucy":
                return force_card(cards, Houndmaster)
            elif PackLeader in minion_types:
                return force_card(cards, PackLeader)
            else:
                return cards[0]

        def select_gain_card(self, cards: List['MonsterCard']) -> 'MonsterCard':
            minion_types = [type(card) for card in cards]
            if KindlyGrandmother in minion_types:
                return force_card(cards, KindlyGrandmother)
            else:
                return cards[0]

    def test_yshaarj_is_combat_summon(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", YShaarj())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 2)
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_2.upgrade_tavern()
        tavern.combat_step()
        tavern.randomizer = self.TestYShaarjCombatSummonRandomizer()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        player_1.hero_power()
        self.assertCardListEquals(player_1.in_play, [PackLeader])
        self.assertCardListEquals(player_2.in_play, [Houndmaster, Houndmaster])
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 37)
        self.assertCardListEquals(player_1.in_play, [PackLeader])
        self.assertCardListEquals(player_1.hand, [KindlyGrandmother])

    def test_bigfernal(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([Bigfernal, Imprisoner, FreedealingGambler, DeadlySpore, DeadlySpore])
        tavern.buying_step()
        for _ in range(2):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        for _ in range(3):
            player_2.purchase(StoreIndex(2))
            player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Bigfernal, Imprisoner])
        self.assertCardListEquals(player_2.in_play, [FreedealingGambler, DeadlySpore, DeadlySpore])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 40)
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)

    def test_champion_of_yshaarj(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer(
            [DragonspawnLieutenant, ChampionOfYShaarj, PartyElemental, HangryDragon, HangryDragon])
        tavern.buying_step()
        for _ in range(2):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        for _ in range(3):
            player_2.purchase(StoreIndex(2))
            player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [DragonspawnLieutenant, ChampionOfYShaarj])
        self.assertCardListEquals(player_2.in_play, [PartyElemental, HangryDragon, HangryDragon])
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 40)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)

    def test_mythrax_the_unraveler(self):
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
        tavern.randomizer = RepeatedCardForcer([MythraxTheUnraveler, AlleyCat])
        tavern.buying_step()
        tavern.randomizer = self.TestLightfangEnforcerRandomizer()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [Amalgam, MurlocTidecaller, DragonspawnLieutenant, VulgarHomunculus,
                                                     ScavengingHyena, KaboomBot, MythraxTheUnraveler])
        self.assertEqual(player_1.in_play[6].attack, player_1.in_play[6].base_attack + 6)
        self.assertEqual(player_1.in_play[6].health, player_1.in_play[6].base_health + 12)

    def test_nzoth(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", NZoth())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        self.assertCardListEquals(player_1.in_play, [FishOfNZoth])

    def test_mrrgglton_can_duplicate_hero(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestMrrggltonRatKingRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", SirFinleyMrrgglton())
        player_2 = tavern.add_player_with_hero("lucy", TheRatKing())
        tavern.buying_step()
        self.assertEqual(type(player_1.hero), SirFinleyMrrgglton)
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.hero), TheRatKing)

    def test_soul_devourer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([Imprisoner, SoulDevourer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Imprisoner])
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        self.assertCardListEquals(player_1.in_play, [SoulDevourer])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 3)
        self.assertEqual(player_1.coins, 4)

    def test_brann_doesnt_double_soul_devourer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([Imprisoner, BrannBronzebeard, SoulDevourer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [Imprisoner, BrannBronzebeard])
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0), [BoardIndex(0)])
        print(player_1.in_play)
        self.assertCardListEquals(player_1.in_play, [BrannBronzebeard, SoulDevourer])
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 3)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 3)

    def test_arm_of_the_empire(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([HangryDragon, HangryDragon, TwilightEmissary, ArmOfTheEmpire])
        tavern.buying_step()
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(3))
        player_1.summon_from_hand(HandIndex(0))
        player_1.purchase(StoreIndex(2))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [ArmOfTheEmpire, TwilightEmissary])
        self.assertCardListEquals(player_2.in_play, [HangryDragon, HangryDragon])
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 34)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 2)

    def test_overlord_saurfang(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", OverlordSaurfang())
        player_2 = tavern.add_player_with_hero("lucy")
        for _ in range(4):
            tavern.buying_step()
            tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        player_1.purchase(StoreIndex(0))
        player_1.purchase(StoreIndex(0))
        self.assertEqual(player_1.hand[0].attack, player_1.hand[0].base_attack + 6)
        self.assertEqual(player_1.hand[1].attack, player_1.hand[1].base_attack)

    def test_xyrella(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Xyrella())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat])
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertEqual(player_1.hand[0].attack, 2)
        self.assertEqual(player_1.hand[0].health, 2)

    def test_voljin_2_store(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Voljin())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat, VulgarHomunculus])
        tavern.buying_step()
        player_1.hero_power(store_index=StoreIndex(0))
        player_1.hero_power(store_index=StoreIndex(1))
        self.assertEqual(player_1.store[0].attack, player_1.store[1].base_attack)
        self.assertEqual(player_1.store[0].health, player_1.store[1].base_health)
        self.assertEqual(player_1.store[1].attack, player_1.store[0].base_attack)
        self.assertEqual(player_1.store[1].health, player_1.store[0].base_health)

    def test_voljin_2_board(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Voljin())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([MurlocTidehunter])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(0))
        player_1.hero_power(board_index=BoardIndex(1))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[0].base_health)

    def test_voljin_board_store(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Voljin())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([MicroMachine, VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(0))
        player_1.hero_power(store_index=StoreIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.store[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.store[0].base_health)
        self.assertEqual(player_1.store[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.store[0].health, player_1.in_play[0].base_health)

    def test_voljin_store_board(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Voljin())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([MicroMachine, VulgarHomunculus])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(store_index=StoreIndex(0))
        player_1.hero_power(board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.store[0].base_attack)
        self.assertEqual(player_1.in_play[0].health, player_1.store[0].base_health)
        self.assertEqual(player_1.store[0].attack, player_1.in_play[0].base_attack)
        self.assertEqual(player_1.store[0].health, player_1.in_play[0].base_health)

    def test_tickatus(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        for i in range(4):
            for _ in range(3):
                tavern.buying_step()
                tavern.combat_step()
            tavern.buying_step()
            self.assertEqual(len(player_1.hero.discover_queue[0]), 3)
            self.assertTrue(all(spell.darkmoon_prize_tier == i + 1 for spell in player_1.hero.discover_queue[0]))
            player_1.hero_select_discover(DiscoverIndex(0))
            self.assertEqual(len(player_1.hero.discover_queue), 0)
            self.assertEqual(player_1.spells[i].darkmoon_prize_tier, i + 1)
            tavern.combat_step()

    def test_gacha_gift(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(GachaGift())  # I know, this is cheating
        player_1.play_spell(SpellIndex(0))
        self.assertTrue(all(card.tier == 1 for card in player_1.discover_queue[0]))

    def test_might_of_stormwind(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(MightOfStormwind())
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 3)

    def test_pocket_change(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(PocketChange())
        player_1.play_spell(SpellIndex(0))
        self.assertCardListEquals(player_1.spells, [GoldCoin, GoldCoin])

    def test_rocking_and_rolling(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(RockingAndRolling())
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(player_1.free_refreshes, 3)

    def test_new_recruit(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(NewRecruit())
        self.assertEqual(len(player_1.store), 3)
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(len(player_1.store), 4)
        player_1.reroll_store()
        self.assertEqual(len(player_1.store), 4)

    def test_the_good_stuff(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(TheGoodStuff())
        player_1.play_spell(SpellIndex(0))
        for card in player_1.store:
            self.assertEqual(card.health, card.base_health + 2)
        player_1.reroll_store()
        for card in player_1.store:
            self.assertEqual(card.health, card.base_health + 2)

    def test_evolving_tavern(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.buying_step()
        player_1.gain_spell(EvolvingTavern())
        expected_tiers = [card.tier + 1 for card in player_1.store]
        player_1.play_spell(SpellIndex(0))
        self.assertListEqual([card.tier for card in player_1.store], expected_tiers)

    def test_great_deal(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(GreatDeal())
        player_1.play_spell(SpellIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        self.assertEqual(player_1.tavern_upgrade_cost, 1)

    def test_gruul_rules(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(GruulRules())
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)

    def test_on_the_house(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.buying_step()
        player_1.gain_spell(OnTheHouse())
        player_1.play_spell(SpellIndex(0))
        self.assertTrue(all(card.tier == 4 for card in player_1.discover_queue[0]))

    def test_the_bouncer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(TheBouncer())
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 5)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 5)
        self.assertTrue(player_1.in_play[0].taunt)

    def test_time_thief(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([MurlocWarleader, Goldgrubber, VulgarHomunculus, AlleyCat, AlleyCat])
        tavern.buying_step()
        for _ in range(3):
            player_2.purchase(StoreIndex(0))
            player_2.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.gain_spell(TimeThief())
        player_1.play_spell(SpellIndex(0))
        warband_types = [MurlocWarleader, Goldgrubber, VulgarHomunculus]
        for card in player_1.discover_queue[0]:
            self.assertIn(type(card), warband_types)

    def test_the_unlimited_coin(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(TheUnlimitedCoin())
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(player_1.spells, [])
        self.assertEqual(player_1.coins, 4)
        tavern.combat_step()
        self.assertCardListEquals(player_1.spells, [TheUnlimitedCoin])

    def test_branns_blessing(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([AlleyCat, MurlocTidehunter])
        tavern.buying_step()
        player_1.gain_spell(BrannsBlessing())
        player_1.play_spell(SpellIndex(0))
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, TabbyCat])
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [AlleyCat, TabbyCat, TabbyCat, MurlocTidehunter, MurlocScout])

    class TestAllThatGlittersRandomizer(DefaultRandomizer):
        def select_from_store(self, store: List['MonsterCard']) -> 'MonsterCard':
            return store[0]

    def test_all_that_glitters(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = self.TestAllThatGlittersRandomizer()
        tavern.buying_step()
        player_1.gain_spell(AllThatGlitters())
        player_1.play_spell(SpellIndex(0))
        self.assertTrue(player_1.store[0].golden)
        self.assertFalse(player_1.store[1].golden)
        self.assertFalse(player_1.store[2].golden)

    def test_BANANAS_randomizer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = self.TestKingMuklaRandomizer()
        tavern.buying_step()
        player_1.gain_spell(Bananas())
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(player_1.hand_size(), 10)
        for spell in player_1.spells:
            self.assertEqual(type(spell), Banana)

    def test_buy_the_holy_light(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(BuyTheHolyLight())
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertTrue(player_1.in_play[0].divine_shield)

    def test_im_still_just_a_rat_in_a_cage(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(ImStillJustARatInACage())
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack * 2)

    def test_repeat_customer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(RepeatCustomer())
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(len(player_1.in_play), 0)
        self.assertCardListEquals(player_1.hand, [DragonspawnLieutenant])
        self.assertEqual(player_1.hand[0].attack, player_1.hand[0].base_attack + 2)
        self.assertEqual(player_1.hand[0].health, player_1.hand[0].base_health + 2)

    def test_top_shelf(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(TopShelf())
        player_1.play_spell(SpellIndex(0))
        self.assertTrue(all(card.tier == 6 for card in player_1.discover_queue[0]))

    def test_ice_block_prize(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(GainIceBlock())
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(type(player_1.secrets[0]), BaseSecret.IceBlock)
        player_1.health = 1
        player_2.purchase(StoreIndex(0))
        player_2.summon_from_hand(HandIndex(0))
        self.assertFalse(player_1.hero.give_immunity)
        tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([VulgarHomunculus])
        tavern.buying_step()
        self.assertFalse(player_1.dead)
        self.assertEqual(player_1.health, 1)
        self.assertEqual(len(player_1.secrets), 0)
        self.assertTrue(player_1.hero.give_immunity)
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertEqual(player_1.health, 1)
        tavern.combat_step()
        self.assertFalse(player_1.hero.give_immunity)

    def test_training_session(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(TrainingSession())
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(len(player_1.hero.discover_queue[0]), 3)
        self.assertTrue(all(issubclass(type(item), Hero) for item in player_1.hero.discover_queue[0]))
        self.assertEqual(type(player_1.hero), Tickatus)
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertNotEqual(type(player_1.hero), Tickatus)

    def test_training_session_choose_millhouse(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestMrrggltonMillhouseRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(TrainingSession())
        player_1.play_spell(SpellIndex(0))
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.hero), MillhouseManastorm)
        self.assertEqual(player_1.minion_cost, 2)
        self.assertEqual(player_1.refresh_store_cost, 2)
        self.assertEqual(player_1.tavern_upgrade_cost, 5)

    class TestTrainingSessionPatchwerkRandomizer(DefaultRandomizer):
        def select_hero(self, hero_pool: List['Hero']) -> 'Hero':
            if PatchWerk in [type(hero) for hero in hero_pool]:
                return [hero for hero in hero_pool if isinstance(hero, PatchWerk)][0]
            else:
                return hero_pool[0]

    def test_training_session_choose_patchwerk(self):
        tavern = Tavern(restrict_types=False)
        tavern.randomizer = self.TestTrainingSessionPatchwerkRandomizer()
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(TrainingSession())
        player_1.play_spell(SpellIndex(0))
        player_1.hero_select_discover(DiscoverIndex(0))
        self.assertEqual(type(player_1.hero), PatchWerk)
        self.assertEqual(player_1.health, 40)

    def test_gain_argent_braggart(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(GainArgentBraggart())
        self.assertCardListEquals(player_1.hand, [ArgentBraggart])
        self.assertCardListEquals(player_1.spells, [])

    def test_fresh_tab(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.buying_step()
        for _ in range(3):
            player_1.purchase(StoreIndex(0))
        player_1.gain_spell(FreshTab())
        self.assertEqual(player_1.coins, 1)
        player_1.play_spell(SpellIndex(0))
        self.assertEqual(player_1.coins, 10)

    def test_friends_and_family_discount(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.gain_spell(FriendsAndFamilyDiscount())
        player_1.play_spell(SpellIndex(0))
        player_1.purchase(StoreIndex(0))
        self.assertEqual(player_1.coins, 2)

    def test_give_a_dog_a_bone(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([MurlocTidecaller])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(GiveADogABone())
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 10)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 10)
        self.assertTrue(player_1.in_play[0].divine_shield)
        self.assertTrue(player_1.in_play[0].windfury)

    def test_open_bar(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(OpenBar())
        player_1.play_spell(SpellIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        for _ in range(5):
            player_1.reroll_store()
        self.assertEqual(player_1.coins, 4)
        player_1.reroll_store()
        self.assertEqual(player_1.coins, 3)

    def test_raise_the_stakes(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(RaiseTheStakes())
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(len(player_1.in_play), 0)
        self.assertCardListEquals(player_1.hand, [DragonspawnLieutenant])
        self.assertTrue(player_1.hand[0].golden)

    def test_big_winner(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", Tickatus())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        player_1.gain_spell(BigWinner())
        player_1.play_spell(SpellIndex(0))
        for i in range(3):
            self.assertEqual(len(player_1.hero.discover_queue), 3 - i)
            self.assertEqual(len(player_1.hero.discover_queue[0]), 3)
            self.assertTrue(all(spell.darkmoon_prize_tier == i + 1 for spell in player_1.hero.discover_queue[0]))
            player_1.hero_select_discover(DiscoverIndex(0))
            print(player_1.spells)
            self.assertEqual(player_1.spells[i].darkmoon_prize_tier, i + 1)

    def test_blood_gem(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([DragonspawnLieutenant])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.gain_spell(BloodGem())
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)

    def test_death_speaker_blackthorn(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", DeathSpeakerBlackthorn())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.buying_step()
        tavern.combat_step()
        tavern.buying_step()
        player_1.upgrade_tavern()
        self.assertCardListEquals(player_1.spells, [BloodGem, BloodGem])

    def test_gruff_runetotem(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", GuffRunetotem())
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer(
            [DragonspawnLieutenant, FreedealingGambler, Khadgar, CaveHydra, KangorsApprentice, GoldrinnTheGreatWolf])
        for i in [0, 3]:
            tavern.buying_step()
            for _ in range(3):
                player_1.purchase(StoreIndex(i))
                player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        tavern.buying_step()
        player_1.hero_power()
        self.assertCardListEquals(player_1.in_play,
                                  [DragonspawnLieutenant, FreedealingGambler, Khadgar, CaveHydra, KangorsApprentice,
                                   GoldrinnTheGreatWolf])
        for card in player_1.in_play:
            self.assertEqual(card.attack, card.base_attack + 1)
            self.assertEqual(card.health, card.base_health + 1)

    def test_mutanus_the_devourer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong", MutanusTheDevourer())
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([MurlocTidehunter])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.hero_power(board_index=BoardIndex(1))
        self.assertCardListEquals(player_1.in_play, [MurlocTidehunter])
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.coins, 1)

    def test_razorfen_geomancer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([RazorfenGeomancer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem])

    def test_sun_bacon_relaxer(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        tavern.randomizer = RepeatedCardForcer([SunBaconRelaxer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.spells, [])
        player_1.sell_minion(BoardIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem, BloodGem])

    def test_prophet_of_the_boar(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = RepeatedCardForcer([ProphetOfTheBoar, SunBaconRelaxer, SunBaconRelaxer])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.spells, [])
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem])
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem])

    def test_tough_tusk(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 2)
        tavern.randomizer = RepeatedCardForcer([SunBaconRelaxer, ToughTusk])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem, BloodGem])
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem])
        player_2.purchase(StoreIndex(1))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [ToughTusk])
        self.assertFalse(player_1.in_play[0].divine_shield)
        self.assertCardListEquals(player_2.in_play, [ToughTusk])
        tavern.combat_step()
        tavern.buying_step()
        self.assertFalse(player_1.in_play[0].divine_shield)
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 36)

    def test_golden_tough_tusk(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([SunBaconRelaxer, ToughTusk, ToughTusk, ToughTusk, Maexxna, Maexxna])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem, BloodGem])
        tavern.combat_step()
        tavern.buying_step()
        for _ in range(3):
            player_1.purchase(StoreIndex(1))
        player_1.summon_from_hand(HandIndex(0))
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem, TripleRewardCard])
        player_2.purchase(StoreIndex(4))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [ToughTusk])
        self.assertTrue(player_1.in_play[0].divine_shield)
        self.assertCardListEquals(player_2.in_play, [Maexxna])
        tavern.combat_step()
        tavern.buying_step()
        self.assertTrue(player_1.in_play[0].divine_shield)
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 32)

    def test_bannerboar(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([SunBaconRelaxer, Bannerboar, SunBaconRelaxer, SunBaconRelaxer])
        tavern.buying_step()
        for _ in range(2):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        tavern.buying_step()
        player_1.purchase(StoreIndex(2))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [SunBaconRelaxer, Bannerboar, SunBaconRelaxer])
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)
        self.assertEqual(player_1.in_play[2].attack, player_1.in_play[2].base_attack + 1)
        self.assertEqual(player_1.in_play[2].health, player_1.in_play[2].base_health + 1)

    def test_bristleback_brute(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([SunBaconRelaxer, BristlebackBrute])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.sell_minion(BoardIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem, BloodGem])
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 3)
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 4)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 4)

    def test_thorncaller(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 3)
        tavern.randomizer = RepeatedCardForcer([Thorncaller])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.spells, [BloodGem])

    def test_dynamic_duo(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([RazorfenGeomancer, DynamicDuo])
        tavern.buying_step()
        for i in range(2):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 1)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 1)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack + 1)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health + 1)

    def test_groundshaker(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([SunBaconRelaxer, Khadgar, Groundshaker, MalGanis, MalGanis])
        tavern.buying_step()
        for i in range(3):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            if i == 0:
                player_1.sell_minion(BoardIndex(0))
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(1))
        self.assertCardListEquals(player_1.in_play, [Khadgar, Groundshaker])
        player_2.purchase(StoreIndex(3))
        player_2.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_2.in_play, [MalGanis])
        tavern.combat_step()
        self.assertEqual(player_1.health, 40)
        self.assertEqual(player_2.health, 40)

    def test_hexruin_marauder(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 4)
        tavern.randomizer = RepeatedCardForcer([HexruinMarauder, AlleyCat, AlleyCat, MurlocTidehunter, AlleyCat])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play, [HexruinMarauder])
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 3)
        tavern.buying_step()
        for _ in range(3):
            player_1.purchase(StoreIndex(1))
            player_1.summon_from_hand(HandIndex(0))
        self.assertCardListEquals(player_1.in_play,
                                  [HexruinMarauder, AlleyCat, TabbyCat, AlleyCat, TabbyCat, MurlocTidehunter,
                                   MurlocScout])
        tavern.combat_step()
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 3)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 3)

    def test_agamaggan_the_great_boar(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("lucy")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = RepeatedCardForcer([RazorfenGeomancer, AgamagganTheGreatBoar])
        tavern.buying_step()
        for i in range(2):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(0))
        self.assertEqual(player_1.in_play[0].attack, player_1.in_play[0].base_attack + 2)
        self.assertEqual(player_1.in_play[0].health, player_1.in_play[0].base_health + 2)
        self.assertEqual(player_1.in_play[1].attack, player_1.in_play[1].base_attack)
        self.assertEqual(player_1.in_play[1].health, player_1.in_play[1].base_health)

    class TestAggemThorncurseRandomizer(DefaultRandomizer):
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
                return force_card(cards, KaboomBot)
            elif round_number == 11:
                return force_card(cards, Scallywag)
            else:
                return cards[0]

    def test_aggem_thorncurse(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe", DeathSpeakerBlackthorn())
        player_2 = tavern.add_player_with_hero("Donald")
        self.upgrade_to_tier(tavern, 5)
        tavern.randomizer = self.TestAggemThorncurseRandomizer()
        for _ in range(6):
            tavern.buying_step()
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
            tavern.combat_step()
        tavern.randomizer = RepeatedCardForcer([AggemThorncurse])
        tavern.buying_step()
        player_1.purchase(StoreIndex(0))
        player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [MurlocTidecaller, DragonspawnLieutenant, VulgarHomunculus,
                                                     ScavengingHyena, KaboomBot, Scallywag, AggemThorncurse])
        self.assertEqual(len(player_1.spells), 8)
        player_1.play_spell(SpellIndex(0), board_index=BoardIndex(6))
        for i in range(6):
            self.assertEqual(player_1.in_play[i].attack, player_1.in_play[i].base_attack + 1)
            self.assertEqual(player_1.in_play[i].health, player_1.in_play[i].base_health + 1)
        self.assertEqual(player_1.in_play[6].attack, player_1.in_play[6].base_attack + 2)
        self.assertEqual(player_1.in_play[6].health, player_1.in_play[6].base_health + 2)

    def test_charlga(self):
        tavern = Tavern(restrict_types=False)
        player_1 = tavern.add_player_with_hero("Joe")
        player_2 = tavern.add_player_with_hero("Donald")
        self.upgrade_to_tier(tavern, 6)
        tavern.randomizer = RepeatedCardForcer([MurlocTidehunter, AgamagganTheGreatBoar, Charlga])
        tavern.buying_step()
        for _ in range(3):
            player_1.purchase(StoreIndex(0))
            player_1.summon_from_hand(HandIndex(0))
        tavern.combat_step()
        self.assertCardListEquals(player_1.in_play, [MurlocTidehunter, MurlocScout, AgamagganTheGreatBoar, Charlga])
        for i in range(3):
            self.assertEqual(player_1.in_play[i].attack, player_1.in_play[i].base_attack + 2)
            self.assertEqual(player_1.in_play[i].health, player_1.in_play[i].base_health + 2)
        self.assertEqual(player_1.in_play[3].attack, player_1.in_play[3].base_attack)
        self.assertEqual(player_1.in_play[3].health, player_1.in_play[3].base_health)


if __name__ == '__main__':
    unittest.main()
