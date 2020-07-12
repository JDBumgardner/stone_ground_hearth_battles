import unittest

from hearthstone.card_pool import *
from hearthstone.cards import Card
from hearthstone.combat import WarParty, fight_boards
from hearthstone.hero_pool import *
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
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [DragonspawnLieutenant()]
        jeremys_war_party.board = [RighteousProtector(), RabidSaurolisk()]
        fight_boards(dianas_war_party, jeremys_war_party, self.TauntTestRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_deathrattle_summon(self):
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [MechaRoo(), DragonspawnLieutenant()]
        jeremys_war_party.board = [DragonspawnLieutenant(), ScavengingHyena()]
        fight_boards(dianas_war_party, jeremys_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_scavenging_hyena(self):
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
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
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [DragonspawnLieutenant(), DragonspawnLieutenant()]
        jeremys_war_party.board = [FiendishServant(), FiendishServant()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_mech_roo(self):
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [MechaRoo()]
        jeremys_war_party.board = [TabbyCat()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_selfless_hero(self):
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [SelflessHero(), TabbyCat()]
        jeremys_war_party.board = [RighteousProtector()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_red_whelp(self):
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [RedWhelp()]
        jeremys_war_party.board = [RighteousProtector()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_harvest_golem(self):
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [HarvestGolem()]
        jeremys_war_party.board = [DragonspawnLieutenant(), MurlocTidecaller()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_kaboom_bot(self):
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [KaboomBot()]
        jeremys_war_party.board = [MurlocTidehunter(), MurlocScout()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_kindly_grandmother(self):
        diana = Player.new_player_with_hero(None, "Diana")
        jeremy = Player.new_player_with_hero(None, "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [KindlyGrandmother()]
        jeremys_war_party.board = [MechaRoo()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_glyph_guardian(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AlleyCat(), TabbyCat(), AlleyCat(), TabbyCat()]
        ethans_war_party.board = [GlyphGuardian()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(ethans_war_party.board[0].attack, 8)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_imprisoner(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MetaltoothLeaper()]
        ethans_war_party.board = [Imprisoner()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)

    def test_murloc_warleader(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MurlocScout(), MurlocWarleader()]
        ethans_war_party.board = [MamaBear()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adams_war_party.board[0].attack, 1)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_scallywag(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Scallywag(), DragonspawnLieutenant()]
        ethans_war_party.board = [VulgarHomunculus(), DamagedGolem()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_unstable_ghoul(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [UnstableGhoul()]
        ethans_war_party.board = [RabidSaurolisk(), WrathWeaver()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_unstable_ghoul_friendly_fire(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [UnstableGhoul(), WrathWeaver(), WrathWeaver(), WrathWeaver(), WrathWeaver()]
        ethans_war_party.board = [RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_rat_pack(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RatPack()]
        ethans_war_party.board = [DragonspawnLieutenant()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 38)

    def test_buffed_rat_pack(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        rat_pack = RatPack()
        rat_pack.attack += 1
        adams_war_party.board = [rat_pack]
        ethans_war_party.board = [DragonspawnLieutenant(), WrathWeaver(), WrathWeaver(), WrathWeaver()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_rat_pack_vs_unstable_ghoul(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        rat_pack = RatPack()
        rat_pack.attack += 1
        unstable_ghoul = UnstableGhoul()
        unstable_ghoul.attack += 1
        adams_war_party.board = [rat_pack]
        ethans_war_party.board = [unstable_ghoul]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        # TODO Jarett: outcome of fight depends on who is attacker and who is defender. What is supposed to happen?

    def test_deathwing(self):
        adam = Player.new_player_with_hero(None, "Adam", Deathwing())
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RatPack()]
        ethans_war_party.board = [DragonspawnLieutenant(), DragonspawnLieutenant(), DragonspawnLieutenant(),
                                  DragonspawnLieutenant(), DragonspawnLieutenant()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_board_size_ignores_dead(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RighteousProtector(), RighteousProtector(), RighteousProtector(), RighteousProtector(), RighteousProtector(), RighteousProtector(), RighteousProtector()]
        ethans_war_party.board = [RighteousProtector(), RighteousProtector(), RighteousProtector(), RighteousProtector(), RighteousProtector(), RighteousProtector(), RatPack()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 37)
        self.assertEqual(ethan.health, 40)

    def test_arcane_cannon(self): # TODO Jarett, does the arcane canon
        # TODO Jacob Finish this test once Jarett helps us figure out the order of operations
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RatPack()]
        ethans_war_party.board = [RatPack()]
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_monstrous_macaw(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MonstrousMacaw(), RighteousProtector(), RatPack()]
        ethans_war_party.board = [RighteousProtector()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 32)

    def test_mechano_egg(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MechanoEgg()]
        ethans_war_party.board = [RighteousProtector(), RighteousProtector(), RighteousProtector(),
                                  RighteousProtector()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_spawn_of_nzoth(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SpawnOfNzoth(), PogoHopper()]
        ethans_war_party.board = [VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    class DeflectOBotRandomizer(DefaultRandomizer):
        def select_enemy_minion(self, enemy_minions: List[Card]) -> Card:
            harvest_golem = [card for card in enemy_minions if type(card) is HarvestGolem]
            return harvest_golem[0]

    def test_deflect_o_bot(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [DeflectOBot(), HarvestGolem()]
        ethans_war_party.board = [KaboomBot()]
        fight_boards(adams_war_party, ethans_war_party, self.DeflectOBotRandomizer())
        self.assertEqual(adams_war_party.board[0].divine_shield, True)
        self.assertEqual(adams_war_party.board[0].attack, 4)
        self.assertEqual(adams_war_party.board[1].dead, True)
        self.assertEqual(adams_war_party.board[2].attack, 2)

    def test_imp_gang_boss(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ImpGangBoss()]
        ethans_war_party.board = [ArcaneCannon(), ArcaneCannon()]
        fight_boards(adams_war_party, ethans_war_party, self.DeflectOBotRandomizer())
        self.assertTrue(adams_war_party.board[0].dead)
        self.assertTrue(adams_war_party.board[2].dead)
        self.assertFalse(adams_war_party.board[1].dead)
        self.assertTrue(isinstance(adams_war_party.board[1], Imp))
        self.assertTrue(isinstance(adams_war_party.board[2], Imp))

if __name__ == '__main__':
    unittest.main()
