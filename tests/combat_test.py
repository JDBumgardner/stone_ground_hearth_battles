import unittest
from typing import List

from hearthstone.card_pool import *
from hearthstone.cards import Card
from hearthstone.combat import WarParty, fight_boards
from hearthstone.player import Player
from hearthstone.randomizer import DefaultRandomizer


class CombatTests(unittest.TestCase):
    class TauntTestRandomizer(DefaultRandomizer):
        def select_attack_target(self, defenders: List[Card]) -> Card:
            target = [card for card in defenders if type(card) is not RighteousProtector]
            if target:
                return target[0]
            else:
                return defenders[0]

    def test_taunt(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [DragonspawnLieutenant()]
        jeremys_war_party.board = [RighteousProtector(), RabidSaurolisk()]
        fight_boards(dianas_war_party, jeremys_war_party, self.TauntTestRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_deathrattle_summon(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [MechaRoo(), DragonspawnLieutenant()]
        jeremys_war_party.board = [DragonspawnLieutenant(), ScavengingHyena()]
        fight_boards(dianas_war_party, jeremys_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_scavenging_hyena(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [DragonspawnLieutenant(), DragonspawnLieutenant()]
        jeremys_war_party.board = [ScavengingHyena(), ScavengingHyena()]
        fight_boards(dianas_war_party, jeremys_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    class FiendishServantRandomizer(DefaultRandomizer):
        def select_attack_target(self, defenders: List[Card]) -> Card:
            return defenders[-1]

    def test_fiendish_servant(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [DragonspawnLieutenant(), DragonspawnLieutenant()]
        jeremys_war_party.board = [FiendishServant(), FiendishServant()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_mech_roo(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [MechaRoo()]
        jeremys_war_party.board = [TabbyCat()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_selfless_hero(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [SelflessHero(), TabbyCat()]
        jeremys_war_party.board = [RighteousProtector()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_red_whelp(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [RedWhelp()]
        jeremys_war_party.board = [RighteousProtector()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_harvest_golem(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [HarvestGolem()]
        jeremys_war_party.board = [DragonspawnLieutenant(), MurlocTidecaller()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_kaboom_bot(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [KaboomBot()]
        jeremys_war_party.board = [MurlocTidehunter(), MurlocScout()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_kindly_grandmother(self):
        diana = Player(None, "Diana")
        jeremy = Player(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [KindlyGrandmother()]
        jeremys_war_party.board = [MechaRoo()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_glyph_guardian(self):
        adam = Player(None, "Adam")
        ethan = Player(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AlleyCat(), TabbyCat(), AlleyCat(), TabbyCat()]
        ethans_war_party.board = [GlyphGuardian()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(ethans_war_party.board[0].attack, 8)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_imprisoner(self):
        adam = Player(None, "Adam")
        ethan = Player(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MetaltoothLeaper()]
        ethans_war_party.board = [Imprisoner()]
        fight_boards( adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)

    def test_murloc_warleader(self):
        adam = Player(None, "Adam")
        ethan = Player(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MurlocScout(), MurlocWarleader()]
        ethans_war_party.board = [MamaBear()]
        fight_boards(adams_war_party, ethans_war_party,DefaultRandomizer())
        self.assertEqual(adams_war_party.board[0].attack, 1)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_scallywag(self):
        adam = Player(None, "Adam")
        ethan = Player(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Scallywag(), DragonspawnLieutenant()]
        ethans_war_party.board = [VulgarHomunculus(), DamagedGolem()]
        fight_boards(adams_war_party, ethans_war_party,  DefaultRandomizer())
        self.assertEqual(ethan.health, 40)
        self.assertEqual(adam.health, 40)

if __name__ == '__main__':
    unittest.main()
