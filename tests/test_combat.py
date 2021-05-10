import random
import unittest
from collections import deque

import logging

from hearthstone.simulator.core.adaptations import AdaptBuffs
from hearthstone.simulator.core.card_graveyard import *
from hearthstone.simulator.core.card_pool import *
from hearthstone.simulator.core.combat import WarParty, fight_boards
from hearthstone.simulator.core.hero_pool import *
from hearthstone.simulator.core.player import Player
from hearthstone.simulator.core.randomizer import DefaultRandomizer
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.testing.battlegrounds_test_case import BattleGroundsTestCase


class CombatTests(BattleGroundsTestCase):
    def assertCardListEquals(self, cards, expected, msg=None):
        self.assertListEqual([type(card) for card in cards], expected, msg=msg)

    def test_taunt(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [VulgarHomunculus(), AlleyCat(), AlleyCat()]
        jeremys_war_party.board = [VulgarHomunculus(), RabidSaurolisk()]
        fight_boards(dianas_war_party, jeremys_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_deathrattle_summon(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [Imprisoner()]
        jeremys_war_party.board = [CrystalWeaver()]
        fight_boards(dianas_war_party, jeremys_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_scavenging_hyena(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [DragonspawnLieutenant(), DragonspawnLieutenant()]
        jeremys_war_party.board = [ScavengingHyena(), ScavengingHyena()]
        fight_boards(dianas_war_party, jeremys_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    class FiendishServantRandomizer(DefaultRandomizer):
        def select_attack_target(self, defenders: List[MonsterCard]) -> MonsterCard:
            return defenders[-1]

    def test_fiendish_servant(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [DragonspawnLieutenant(), DragonspawnLieutenant()]
        jeremys_war_party.board = [FiendishServant(), FiendishServant()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_selfless_hero(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [SelflessHero(), TabbyCat()]
        jeremys_war_party.board = [FreedealingGambler()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_red_whelp(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [RedWhelp()]
        jeremys_war_party.board = [Scallywag()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 38)

    def test_harvest_golem(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [HarvestGolem()]
        jeremys_war_party.board = [DragonspawnLieutenant(), MurlocTidecaller()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_kaboom_bot(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [KaboomBot()]
        jeremys_war_party.board = [MurlocTidehunter(), MurlocScout()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_kindly_grandmother(self):
        diana = Player.new_player_with_hero(Tavern(), "Diana")
        jeremy = Player.new_player_with_hero(Tavern(), "Jeremy")
        dianas_war_party = WarParty(diana)
        jeremys_war_party = WarParty(jeremy)
        dianas_war_party.board = [KindlyGrandmother()]
        jeremys_war_party.board = [CrystalWeaver()]
        fight_boards(jeremys_war_party, dianas_war_party, DefaultRandomizer())
        self.assertEqual(diana.health, 40)
        self.assertEqual(jeremy.health, 40)

    def test_glyph_guardian(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AlleyCat(), TabbyCat(), AlleyCat(), TabbyCat()]
        ethans_war_party.board = [GlyphGuardian()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(ethans_war_party.board[0].attack, 8)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_imprisoner(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MetaltoothLeaper()]
        ethans_war_party.board = [Imprisoner()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)

    def test_murloc_warleader(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MurlocScout(), MurlocWarleader()]
        ethans_war_party.board = [IronhideRunt()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adams_war_party.board[0].attack, 1)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_scallywag(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Scallywag(), DragonspawnLieutenant()]
        ethans_war_party.board = [VulgarHomunculus(), DamagedGolem()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_unstable_ghoul(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [UnstableGhoul()]
        ethans_war_party.board = [RabidSaurolisk(), FiendishServant()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_unstable_ghoul_friendly_fire(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [UnstableGhoul(), FiendishServant(), FiendishServant(), FiendishServant(),
                                 FiendishServant()]
        ethans_war_party.board = [RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_rat_pack(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RatPack()]
        ethans_war_party.board = [DragonspawnLieutenant()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 38)

    def test_buffed_rat_pack(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        rat_pack = RatPack()
        rat_pack.attack += 1
        adams_war_party.board = [rat_pack]
        ethans_war_party.board = [DragonspawnLieutenant(), FiendishServant(), FiendishServant(), FiendishServant()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_rat_pack_vs_unstable_ghoul(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
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
        adam = Player.new_player_with_hero(Tavern(), "Adam", Deathwing())
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RatPack()]
        ethans_war_party.board = [DragonspawnLieutenant(), DragonspawnLieutenant(), DragonspawnLieutenant(),
                                  DragonspawnLieutenant(), DragonspawnLieutenant()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_board_size_ignores_dead(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        ratpack = RatPack()
        ratpack.golden_transformation([])
        adams_war_party.board = [TwilightEmissary() for _ in range(7)]
        ethans_war_party.board = [TwilightEmissary() for _ in range(6)] + [ratpack]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 35)
        self.assertEqual(ethan.health, 40)

    def test_mechano_egg(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MechanoEgg()]
        ethans_war_party.board = [DeckSwabbie() for _ in range(4)]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_spawn_of_nzoth(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SpawnOfNzoth(), AlleyCat()]
        ethans_war_party.board = [VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    class DeflectOBotRandomizer(DefaultRandomizer):
        def select_enemy_minion(self, enemy_minions: List[MonsterCard]) -> MonsterCard:
            harvest_golem = [card for card in enemy_minions if type(card) is HarvestGolem]
            return harvest_golem[0]

    def test_deflect_o_bot(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
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
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ImpGangBoss()]
        ethans_war_party.board = [RabidSaurolisk(), Rat(), Rat(), Rat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertTrue(isinstance(adams_war_party.board[1], Imp))

    def test_infested_wolf(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [InfestedWolf()]
        ethans_war_party.board = [FreedealingGambler(), DeckSwabbie()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_monstrous_macaw(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MonstrousMacaw(), Imprisoner(), AlleyCat()]
        ethans_war_party.board = [TwilightEmissary(), CapnHoggarr()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adams_war_party.board), 5)

    def test_soul_juggler(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [IronhideRunt()]
        ethans_war_party.board = [Imp(), SoulJuggler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_golden_soul_juggler(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [PackLeader(), PackLeader()]
        soul_juggler = SoulJuggler()
        soul_juggler.golden_transformation([])
        ethans_war_party.board = [Imp(), Imp(), soul_juggler]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 35)
        self.assertEqual(ethan.health, 40)

    def test_khadgar(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [HarvestGolem(), Khadgar()]
        ethans_war_party.board = [RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 34)

    class SneedsOldShredderRandomizer(DefaultRandomizer):
        def select_summon_minion(self, card_types: List['Type']) -> 'Type':
            khadgar = [card_type for card_type in card_types if card_type == Khadgar]
            return khadgar[0]

    def test_khadgar_sneeds_old_shredder(self):
        logging.basicConfig(level=logging.DEBUG)
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        sneeds = SneedsOldShredder()
        sneeds.golden_transformation([])
        nadina = NadinaTheRed()
        nadina.golden_transformation([])
        adams_war_party.board = [sneeds, Khadgar()]
        ethans_war_party.board = [nadina]
        fight_boards(adams_war_party, ethans_war_party, self.SneedsOldShredderRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 18)

    def test_savannah_highmane(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SavannahHighmane()]
        ethans_war_party.board = [IronhideRunt()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 37)
        self.assertEqual(len(adams_war_party.board), 3)

    def test_security_rover(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SecurityRover()]
        ethans_war_party.board = [RabidSaurolisk(), RabidSaurolisk(), RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 38)
        self.assertEqual(len(adams_war_party.board), 3)

    def test_ripsnarl_captain(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [FreedealingGambler(), RipsnarlCaptain()]
        ethans_war_party.board = [IronhideRunt()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 35)

    def test_southsea_captain(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [FreedealingGambler(), SouthseaCaptain()]
        ethans_war_party.board = [TwilightEmissary()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adams_war_party.board[0].attack, 4)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 37)

    def test_sneeds_old_shredder(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SneedsOldShredder()]
        ethans_war_party.board = [BloodsailCannoneer(), BloodsailCannoneer()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertNotEqual(ethan.health, 40)
        self.assertNotEqual(len(adams_war_party.board), 1)
        self.assertTrue(adams_war_party.board[1].legendary)

    def test_bolvar_fireblood(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [BolvarFireblood()]
        bloodsail_cannoneer = BloodsailCannoneer()
        bloodsail_cannoneer.golden_transformation([])
        ethans_war_party.board = [bloodsail_cannoneer]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_drakonid_enforcer(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AnnoyOModule(), DrakonidEnforcer()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_bronze_warden(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SpawnOfNzoth(), BronzeWarden()]
        ironhide_runt = IronhideRunt()
        ironhide_runt.golden_transformation([])
        ethans_war_party.board = [ironhide_runt]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 36)
        self.assertEqual(adams_war_party.board[2].attack, adams_war_party.board[2].base_attack)
        self.assertEqual(adams_war_party.board[2].health, 1)

    def test_golden_bronze_warden(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        bronze_warden = BronzeWarden()
        bronze_warden.golden_transformation([])
        adams_war_party.board = [bronze_warden]
        drakonid_enforcer = DrakonidEnforcer()
        drakonid_enforcer.golden_transformation([])
        ethans_war_party.board = [drakonid_enforcer]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 36)
        self.assertEqual(adams_war_party.board[1].attack, adams_war_party.board[1].base_attack * 2)
        self.assertEqual(adams_war_party.board[1].health, 1)

    def test_replicating_menace(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ReplicatingMenace()]
        ethans_war_party.board = [RatPack(), Rat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adams_war_party.board), 4)

    def test_junkbot(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [KaboomBot(), Junkbot()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_voidlord(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Microbot(), Voidlord()]
        virmen = VirmenSensei()
        virmen.golden_transformation([])
        ethans_war_party.board = [virmen]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_king_bagurgle(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [KingBagurgle(), MurlocScout(), AlleyCat()]
        ironhide_runt = IronhideRunt()
        ironhide_runt.golden_transformation([])
        ethans_war_party.board = [ironhide_runt]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_ghastcoiler(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Ghastcoiler()]
        bloodsail_cannoneer = BloodsailCannoneer()
        bloodsail_cannoneer.golden_transformation([])
        ethans_war_party.board = [bloodsail_cannoneer]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertNotEqual(ethan.health, 40)
        self.assertEqual(len(adams_war_party.board), 3)
        for card in adams_war_party.board:
            self.assertTrue(card.deathrattles)

    def test_dread_admiral_eliza(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [DeckSwabbie(), DreadAdmiralEliza()]
        hoggarr = CapnHoggarr()
        hoggarr.golden_transformation([])
        ethans_war_party.board = [hoggarr]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_goldrinn_the_great_wolf(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [GoldrinnTheGreatWolf(), ScavengingHyena()]
        ironhide_runt = IronhideRunt()
        ironhide_runt.golden_transformation([])
        ethans_war_party.board = [ironhide_runt]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_imp_mama(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ImpMama()]
        ethans_war_party.board = [AlleyCat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertNotEqual(ethan.health, 40)
        self.assertEqual(len(adams_war_party.board), 2)
        self.assertTrue(adams_war_party.board[1].check_type(MONSTER_TYPES.DEMON))
        self.assertTrue(adams_war_party.board[1].taunt)

    def test_nadina_the_red(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [NadinaTheRed(), StewardOfTime()]
        ethans_war_party.board = [KalecgosArcaneAspect()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_the_tide_razor(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TheTideRazor()]
        ethans_war_party.board = [BloodsailCannoneer()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertNotEqual(ethan.health, 40)
        self.assertEqual(len(adams_war_party.board), 4)
        for i in range(1, len(adams_war_party.board)):
            self.assertTrue(adams_war_party.board[i].check_type(MONSTER_TYPES.PIRATE))

    def test_maexxna(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Maexxna()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_herald_of_flame(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [HeraldOfFlame(), Rat(), Rat()]
        ethans_war_party.board = [Robosaur(), UnstableGhoul()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_herald_of_flame_chain(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [HeraldOfFlame(), Rat(), Rat(), Rat(), Rat(), Rat(), Rat()]
        ethans_war_party.board = [MicroMachine(), MicroMachine(), MicroMachine(), MicroMachine(), MicroMachine(),
                                  UnstableGhoul()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 35)

    def test_herald_of_flame_defender(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        herald = HeraldOfFlame()
        herald.taunt = True
        adams_war_party.board = [herald, FreedealingGambler()]
        ethans_war_party.board = [NadinaTheRed(), NadinaTheRed(), NadinaTheRed()]
        for card in ethans_war_party.board:
            card.health -= 1
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 33)
        self.assertEqual(ethan.health, 40)

    def test_ironhide_direhorn(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [IronhideDirehorn(), Rat()]
        ethans_war_party.board = [NadinaTheRed()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 37)

    def test_ironhide_direhorn_defender(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [IronhideDirehorn()]
        ethans_war_party.board = [NadinaTheRed(), NadinaTheRed()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(len(adams_war_party.board), 1)
        self.assertEqual(adam.health, 33)
        self.assertEqual(ethan.health, 40)

    def test_nat_pagle_extreme_angler(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [NatPagleExtremeAngler()]
        ethans_war_party.board = [MechanoEgg()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(adam.hand_size(), 1)

    def test_nat_pagle_defender(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [NatPagleExtremeAngler()]
        ethans_war_party.board = [IronhideRunt(), Rat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(adam.hand_size(), 0)

    def test_mal_ganis(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Imp(), MalGanis()]
        hoggarr = CapnHoggarr()
        hoggarr.golden_transformation([])
        ethans_war_party.board = [hoggarr]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        # self.assertEqual(adams_war_party.board[0].attack, 1)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_old_murkeye(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [OldMurkeye()]
        ethans_war_party.board = [KingBagurgle()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_baron_rivendare(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [KindlyGrandmother(), BaronRivendare()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_macaw_goldrinn(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MonstrousMacaw(), GoldrinnTheGreatWolf()]
        ethans_war_party.board = [FreedealingGambler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertFalse(adams_war_party.board[0].dead)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 30)
        self.assertEqual(adams_war_party.board[0].attack, 10)
        self.assertEqual(adams_war_party.board[1].attack, 9)
        self.assertEqual(adams_war_party.board[1].health, 9)

    def test_macaw_baron(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MonstrousMacaw(), GoldrinnTheGreatWolf(), BaronRivendare()]
        ethans_war_party.board = [FreedealingGambler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertFalse(adams_war_party.board[0].dead)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 25)
        self.assertEqual(adams_war_party.board[0].attack, 15)
        self.assertEqual(adams_war_party.board[1].attack, 14)
        self.assertEqual(adams_war_party.board[1].health, 14)

    def test_zero_attack(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MechanoEgg(), IronhideRunt()]
        ethans_war_party.board = [IronhideRunt()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertFalse(adams_war_party.board[0].dead)
        self.assertEqual(len(adams_war_party.board), 2)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 35)

    def test_yo_ho_ogre(self):
        logging.basicConfig(level=logging.DEBUG)
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [YoHoOgre(), RipsnarlCaptain()]
        ethans_war_party.board = [DeckSwabbie(), TwilightEmissary(), Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_waxrider_togwaggle(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        houndmaster = Houndmaster()
        houndmaster.attack += 1
        adams_war_party.board = [TwilightEmissary(), WaxriderTogwaggle()]
        ethans_war_party.board = [TwilightEmissary(), houndmaster]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_herald_togwaggle(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        robosaur = Robosaur()
        robosaur.golden_transformation([])
        adams_war_party.board = [HeraldOfFlame(), Rat(), Rat(), Rat(), WaxriderTogwaggle()]
        ethans_war_party.board = [MicroMachine(), MicroMachine(), robosaur, UnstableGhoul()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_mama_bear_in_combat(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SavannahHighmane(), MamaBear(), VulgarHomunculus()]
        ethans_war_party.board = [RabidSaurolisk(), NadinaTheRed(), NadinaTheRed(), NadinaTheRed(), NadinaTheRed()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_micro_mummy(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MicroMummy()]
        ethans_war_party.board = [RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_amalgadon(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        amalgadon = Amalgadon()
        for adaptation in valid_adaptations(amalgadon):
            amalgadon.adapt(adaptation())
        self.assertTrue(amalgadon.divine_shield)
        self.assertTrue(amalgadon.windfury)
        self.assertTrue(amalgadon.taunt)
        self.assertTrue(amalgadon.poisonous)
        self.assertEqual(len(amalgadon.deathrattles), 1)
        self.assertEqual(amalgadon.attack, amalgadon.base_attack + 4)
        self.assertEqual(amalgadon.health, amalgadon.base_health + 4)
        runt = IronhideRunt()
        runt.golden_transformation([])
        runt.taunt = True
        adams_war_party.board = [amalgadon]
        ethans_war_party.board = [InfestedWolf(), runt]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_kangors_apprentice(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SecurityRover(), KangorsApprentice()]
        ethans_war_party.board = [Maexxna()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 34)

    def test_zapp_slywick(self):  # TODO I don't think this actually tests everything about this card
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ZappSlywick()]
        ethans_war_party.board = [AlleyCat(), AlleyCat(), MonstrousMacaw(), MechanoEgg()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)

    def test_foe_reaper_4000(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [FoeReaper4000()]
        ethans_war_party.board = [AlleyCat(), MamaBear(), DragonspawnLieutenant(), MamaBear()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 33)

    def test_macaw_multiple_deathrattles(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        amalgadon = Amalgadon()
        amalgadon.adapt(AdaptBuffs.LivingSpores())
        amalgadon.adapt(AdaptBuffs.LivingSpores())
        adams_war_party.board = [MonstrousMacaw(), amalgadon]
        ethans_war_party.board = [RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(len(adams_war_party.board), 6)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 29)

    def test_golden_zapp(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        zapp = ZappSlywick()
        zapp.golden_transformation([])
        adams_war_party.board = [zapp, TwilightEmissary()]
        ethans_war_party.board = [BloodsailCannoneer(), DeckSwabbie(), BloodsailCannoneer(), CapnHoggarr(),
                                  NatPagleExtremeAngler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_seabreaker_goliath(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SeabreakerGoliath(), DeckSwabbie(), DeckSwabbie(), VulgarHomunculus()]
        ethans_war_party.board = [RabidSaurolisk(), CapnHoggarr(), CapnHoggarr(), TwilightEmissary(),
                                  TwilightEmissary()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_siege_breaker(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [VulgarHomunculus(), Siegebreaker()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_crackling_cyclone(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [CracklingCyclone(), TwilightEmissary()]
        ethans_war_party.board = [BloodsailCannoneer(), VulgarHomunculus(), VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_deadly_spore(self):
        adam = Player.new_player_with_hero(Tavern(), "Adam")
        ethan = Player.new_player_with_hero(Tavern(), "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [DeadlySpore()]
        ethans_war_party.board = [KalecgosArcaneAspect()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    class GentleDjinniRandomizer(DefaultRandomizer):
        def select_summon_minion(self, card_types: List['Type']) -> 'Type':
            if Sellemental in card_types:
                return Sellemental
            else:
                return card_types[0]

    def test_gentle_djinni(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [GentleDjinni()]
        ethans_war_party.board = [IronhideRunt()]
        fight_boards(adams_war_party, ethans_war_party, self.GentleDjinniRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(adam.hand_size(), 1)
        self.assertEqual(type(adam.hand[0]), Sellemental)

    def test_djinni_khadgar(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [GentleDjinni(), Khadgar()]
        ethans_war_party.board = [ZappSlywick()]
        fight_boards(adams_war_party, ethans_war_party, self.GentleDjinniRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(adam.hand_size(), 1)
        self.assertEqual(type(adam.hand[0]), Sellemental)

    def test_wildfire_elemental(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [WildfireElemental(), CapnHoggarr(), VulgarHomunculus()]
        ethans_war_party.board = [RabidSaurolisk(), CapnHoggarr(), LieutenantGarr(), CapnHoggarr()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_golden_wildfire_elemental(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        wildfire_elemental = WildfireElemental()
        wildfire_elemental.golden_transformation([])
        garr = LieutenantGarr()
        garr.attack -= 1
        adams_war_party.board = [wildfire_elemental, TwilightEmissary()]
        ethans_war_party.board = [BloodsailCannoneer(), CapnHoggarr(), garr, CapnHoggarr(), AlleyCat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_cave_hydra(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan", Deathwing())
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RabidSaurolisk(), CrystalWeaver(), VulgarHomunculus(), CrystalWeaver()]
        ethans_war_party.board = [CaveHydra(), VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_baron_doesnt_stack(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [UnstableGhoul(), BaronRivendare(), BaronRivendare()]
        ethans_war_party.board = [FreedealingGambler(), DragonspawnLieutenant(), DragonspawnLieutenant(),
                                  DragonspawnLieutenant(), DragonspawnLieutenant(), DragonspawnLieutenant(),
                                  DragonspawnLieutenant()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_illidan_stormrage(self):
        logging.basicConfig(level=logging.DEBUG)
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", IllidanStormrage())
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [GlyphGuardian(), TwilightEmissary(), DeadlySpore()]
        ethans_war_party.board = [BloodsailCannoneer(), Robosaur(), VulgarHomunculus(), VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_hipster_pirate_build(self):
        logging.basicConfig(level=logging.DEBUG)
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ImpMama() for _ in range(7)]
        scally1 = Scallywag()
        scally1.taunt = True
        scally2 = Scallywag()
        scally2.taunt = True
        chad1 = Khadgar()
        chad1.golden_transformation([])
        chad2 = Khadgar()
        chad2.golden_transformation([])
        baron = BaronRivendare()
        baron.golden_transformation([])
        ethans_war_party.board = [scally1, scally2, DreadAdmiralEliza(), chad1, chad2, baron]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        print(adams_war_party.board)
        print(ethans_war_party.board)
        self.assertNotEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_ice_block(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.ICE_BLOCK)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adam.health = 1
        ethans_war_party.board = [Amalgadon()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 1)
        self.assertEqual(len(adam.hero.secrets), 0)
        self.assertTrue(adam.hero.give_immunity)
        self.assertFalse(adam.dead)

    def test_splitting_image(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.SPLITTING_IMAGE)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TwilightEmissary()]
        ethans_war_party.board = [BloodsailCannoneer(), BloodsailCannoneer()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adam.hero.secrets), 0)

    def test_snake_trap(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.SNAKE_TRAP)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AlleyCat()]
        ethans_war_party.board = [MurlocTidehunter() for _ in range(4)]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        print(adams_war_party.board)
        print(ethans_war_party.board)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adam.hero.secrets), 0)

    def test_venomstrike_trap(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.VENOMSTRIKE_TRAP)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MurlocScout()]
        ethans_war_party.board = [AlleyCat(), KalecgosArcaneAspect()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adam.hero.secrets), 0)

    def test_autodefense_matrix(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.AUTODEFENSE_MATRIX)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TabbyCat()]
        ethans_war_party.board = [AlleyCat(), AlleyCat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adam.hero.secrets), 0)

    def test_redemption(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.REDEMPTION)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TwilightEmissary()]
        ethans_war_party.board = [BloodsailCannoneer(), AlleyCat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adam.hero.secrets), 0)

    def test_avenge(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.AVENGE)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TwilightEmissary(), AlleyCat()]
        ethans_war_party.board = [BloodsailCannoneer(), DeckSwabbie(), DeckSwabbie()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adam.hero.secrets), 0)

    def test_illidan_stormrage_triggers_before_red_whelp(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", IllidanStormrage())
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [DeckSwabbie(), RockpoolHunter(), DeckSwabbie()]
        ethans_war_party.board = [DragonspawnLieutenant(), RedWhelp()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_deathwing_buffs_before_illidan_stormrage(self):
        logging.basicConfig(level=logging.DEBUG)
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", IllidanStormrage())
        ethan = tavern.add_player_with_hero("Ethan", Deathwing())
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TwilightEmissary(), TwilightEmissary()]
        ethans_war_party.board = [DeckSwabbie(), DeckSwabbie()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_illidan_one_minion(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", IllidanStormrage())
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [NatPagleExtremeAngler()]
        ethans_war_party.board = [TwilightEmissary(), TwilightEmissary()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adam.hand), 1)

    def test_illidan_triggers_windfury(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", IllidanStormrage())
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [CracklingCyclone(), VulgarHomunculus(), CracklingCyclone()]
        ethans_war_party.board = [TwilightEmissary(), TwilightEmissary(), TwilightEmissary(), TwilightEmissary()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 38)

    def test_acolyte_of_cthun(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [VulgarHomunculus(), TabbyCat()]
        ethans_war_party.board = [AcolyteOfCThun(), AlleyCat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_tormented_ritualist(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [CapnHoggarr(), TwilightEmissary()]
        ethans_war_party.board = [BloodsailCannoneer(), AlleyCat(), TormentedRitualist(), AlleyCat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_warden_of_old(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [WardenOfOld()]
        ethans_war_party.board = [FreedealingGambler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertCardListEquals(adam.spells, [GoldCoin])

    def test_arm_of_the_empire(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TwilightEmissary(), TwilightEmissary(), TwilightEmissary()]
        ethans_war_party.board = [ArmOfTheEmpire(), VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_qiraji_harbinger(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RabidSaurolisk(), CrystalWeaver(), CrystalWeaver()]
        ethans_war_party.board = [AlleyCat(), VulgarHomunculus(), QirajiHarbinger()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_adjacent_minions(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        servant = FiendishServant()
        servant.dead = True
        mummy = MicroMummy()
        hyena = ScavengingHyena()
        hyena.dead = True
        homunculus = VulgarHomunculus()
        homunculus.dead = True
        adams_war_party.board = [AlleyCat(), servant, mummy, MurlocTidecaller()]
        ethans_war_party.board = [RockpoolHunter(), hyena, homunculus, DeckSwabbie()]
        self.assertListEqual([type(card) for card in adams_war_party.adjacent_minions(mummy)],
                             [AlleyCat, MurlocTidecaller])
        self.assertListEqual([type(card) for card in ethans_war_party.adjacent_minions(homunculus)],
                             [RockpoolHunter, DeckSwabbie])

    def test_elistra_the_immortal(self):
        logging.basicConfig(level=logging.DEBUG)
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TwilightEmissary(), TwilightEmissary(), TwilightEmissary(), TwilightEmissary()]
        ethans_war_party.board = [ElistraTheImmortal(), VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)

    def test_elistra_on_attack_triggers(self):
        logging.basicConfig(level=logging.DEBUG)
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        saurolisk = RabidSaurolisk()
        saurolisk.taunt = True
        adams_war_party.board = [IronhideRunt(), IronhideRunt(), CrystalWeaver(), CrystalWeaver(), saurolisk]
        ethans_war_party.board = [TormentedRitualist(), ElistraTheImmortal()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_fish_of_nzoth(self):
        logging.basicConfig(level=logging.DEBUG)
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Imprisoner(), FishOfNZoth()]
        ethans_war_party.board = [CapnHoggarr()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    class TestKhadgarSummonsRandomizer(DefaultRandomizer):
        def select_summon_minion(self, card_types: List['Type']) -> 'Type':
            if VulgarHomunculus in card_types:
                return VulgarHomunculus  # Piloted Shredder, Imp Mama
            elif LieutenantGarr in card_types:
                return LieutenantGarr  # Sneeds Old Shredder, Gentle Djinni
            elif DeckSwabbie in card_types:
                return DeckSwabbie  # The Tide Razor
            elif WardenOfOld in card_types:
                return WardenOfOld  # Ghastcoiler
            else:
                return card_types[0]

    def test_sneeds_old_shredder_khadgar_same_minion(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        sneeds = SneedsOldShredder()
        sneeds.taunt = True
        adams_war_party.board = [sneeds, Khadgar()]
        ethans_war_party.board = [LieutenantGarr(), LieutenantGarr(), LieutenantGarr(), LieutenantGarr()]
        fight_boards(adams_war_party, ethans_war_party, self.TestKhadgarSummonsRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_imp_mama_khadgar_same_minion(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        mama = ImpMama()
        mama.taunt = True
        adams_war_party.board = [mama, Khadgar()]
        garr = LieutenantGarr()
        garr.golden_transformation([])
        ethans_war_party.board = [garr, LieutenantGarr(), LieutenantGarr(), LieutenantGarr()]
        fight_boards(adams_war_party, ethans_war_party, self.TestKhadgarSummonsRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_sneeds_khadgar_same_minion(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        sneeds = SneedsOldShredder()
        sneeds.taunt = True
        adams_war_party.board = [sneeds, Khadgar()]
        ethans_war_party.board = [NadinaTheRed(), NadinaTheRed(), NadinaTheRed()]
        fight_boards(adams_war_party, ethans_war_party, self.TestKhadgarSummonsRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 36)

    def test_djinni_khadgar_same_minion(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [GentleDjinni(), Khadgar()]
        ethans_war_party.board = [LieutenantGarr(), LieutenantGarr(), LieutenantGarr(), LieutenantGarr()]
        fight_boards(adams_war_party, ethans_war_party, self.TestKhadgarSummonsRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_boat_khadgar_same_minion(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        boat = TheTideRazor()
        boat.taunt = True
        adams_war_party.board = [boat, Khadgar()]
        ethans_war_party.board = [LieutenantGarr(), DeckSwabbie(), VulgarHomunculus(), VulgarHomunculus(),
                                  VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, self.TestKhadgarSummonsRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_ghastcoiler_khadgar_same_minion(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        ghastcoiler = Ghastcoiler()
        ghastcoiler.taunt = True
        adams_war_party.board = [ghastcoiler, Khadgar()]
        ethans_war_party.board = [NadinaTheRed(), RabidSaurolisk(), RabidSaurolisk(), RabidSaurolisk(),
                                  RabidSaurolisk(), RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, self.TestKhadgarSummonsRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_splitting_image_keeps_buffs(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.SPLITTING_IMAGE)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        alleycat = AlleyCat()
        alleycat.poisonous = True
        adams_war_party.board = [alleycat]
        ethans_war_party.board = [KalecgosArcaneAspect(), KalecgosArcaneAspect()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_splitting_image_waits(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan", TheGreatAkazamzarak())
        ethan.hero.secrets.append(SECRETS.SPLITTING_IMAGE)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AlleyCat() for _ in range(7)]
        ethans_war_party.board = [AlleyCat() for _ in range(7)]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)

    def test_splitting_image_khadgar(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.SPLITTING_IMAGE)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [LieutenantGarr(), Khadgar()]
        ethans_war_party.board = [RabidSaurolisk(), RabidSaurolisk(), RabidSaurolisk(), RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_autodefense_matrix_waits(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.AUTODEFENSE_MATRIX)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AlleyCat(), AnnoyOModule()]
        ethans_war_party.board = [CracklingCyclone(), DeadlySpore(), DeadlySpore()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adam.hero.secrets), 0)

    def test_venomstrike_trap_waits(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan", TheGreatAkazamzarak())
        ethan.hero.secrets.append(SECRETS.VENOMSTRIKE_TRAP)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [BloodsailCannoneer() for _ in range(7)]
        ethans_war_party.board = [TwilightEmissary() for _ in range(7)]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)

    def test_venomstrike_trap_khadgar(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.VENOMSTRIKE_TRAP)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Khadgar()]
        ethans_war_party.board = [RabidSaurolisk(), RabidSaurolisk(), RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_snake_trap_waits(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan", TheGreatAkazamzarak())
        ethan.hero.secrets.append(SECRETS.SNAKE_TRAP)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AlleyCat() for _ in range(7)]
        ethans_war_party.board = [AlleyCat() for _ in range(7)]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 37)
        self.assertEqual(ethan.health, 40)

    def test_snake_trap_khadgar(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.SNAKE_TRAP)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Khadgar()]
        ethans_war_party.board = [FiendishServant() for _ in range(7)]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_redemption_khadgar(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", TheGreatAkazamzarak())
        ethan = tavern.add_player_with_hero("Ethan")
        adam.hero.secrets.append(SECRETS.REDEMPTION)
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [AlleyCat(), Khadgar()]
        ethans_war_party.board = [IronhideRunt()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_elistra_doesnt_trigger_on_attack_twice(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        glyph_guardian = GlyphGuardian()
        glyph_guardian.health += 1
        adams_war_party.board = [glyph_guardian, VulgarHomunculus()]
        ethans_war_party.board = [RabidSaurolisk(), Siegebreaker(), ElistraTheImmortal()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 29)
        self.assertEqual(ethan.health, 40)

    def test_ring_watcher(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RingWatcher()]
        ethans_war_party.board = [KalecgosArcaneAspect()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_greybough(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam", Greybough())
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        ironhide_runt = IronhideRunt()
        ironhide_runt.golden_transformation([])
        adams_war_party.board = [Imprisoner(), ChampionOfYShaarj()]
        ethans_war_party.board = [ironhide_runt]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_ghoul_scavenger(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        ghoul = UnstableGhoul()
        ghoul.golden_transformation([])
        garr = LieutenantGarr()
        garr.golden_transformation([])
        adams_war_party.board = [ghoul]
        ethans_war_party.board = [garr, TabbyCat(), ScavengingHyena()]  # hyena dies before it is buffed
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_kaboom_bot_in_the_queue(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        kaboom_bot = KaboomBot()
        kaboom_bot.golden_transformation([])
        kaboom_bot.taunt = True
        adams_war_party.board = [TwilightEmissary(), KaboomBot(), KaboomBot()]
        ethans_war_party.board = [kaboom_bot, AlleyCat(), AlleyCat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_red_whelp_in_the_queue(self):
        logging.basicConfig(level=logging.DEBUG)
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        red_whelp = RedWhelp()
        red_whelp.golden_transformation([])
        garr = LieutenantGarr()
        garr.golden_transformation([])
        adams_war_party.board = [ReplicatingMenace(), KindlyGrandmother()]
        ethans_war_party.board = [red_whelp, garr]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_soul_juggler_in_the_queue(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        soul_juggler = SoulJuggler()
        soul_juggler.golden_transformation([])
        adams_war_party.board = [LieutenantGarr(), KindlyGrandmother(), ReplicatingMenace()]
        ethans_war_party.board = [soul_juggler, VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    class TestKaboomBotResolvingBeforeSpawnRandomizer(DefaultRandomizer):
        def select_event_queue(self, queues: List[deque]) -> deque:
            first_queue_minions = [type(card) for card, foe in queues[0]]
            second_queue_minions = [type(card) for card, foe in queues[1]]
            if KaboomBot in first_queue_minions:
                return queues[0]
            elif KaboomBot in second_queue_minions:
                return queues[1]
            else:
                return random.choice(queues)

    def test_kaboom_bot_resolving_before_spawn(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        unstable_ghoul = UnstableGhoul()
        unstable_ghoul.golden_transformation([])
        garr = LieutenantGarr()
        garr.attack += 1
        garr.health += 1
        adams_war_party.board = [unstable_ghoul, KaboomBot(), KaboomBot()]
        ethans_war_party.board = [garr, SpawnOfNzoth(), CapnHoggarr()]
        fight_boards(adams_war_party, ethans_war_party, self.TestKaboomBotResolvingBeforeSpawnRandomizer())
        self.assertEqual(adam.health, 34)
        self.assertEqual(ethan.health, 40)

    def test_double_red_whelp(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RedWhelp(), RedWhelp()]
        ethans_war_party.board = [SelflessHero(), SelflessHero()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 38)

    def test_barrens_blacksmith(self):
        tavern = Tavern()
        adam = tavern.add_player_with_hero("Adam")
        ethan = tavern.add_player_with_hero("Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        blacksmith = BarrensBlacksmith()
        blacksmith.taunt = True
        adams_war_party.board = [blacksmith, AlleyCat()]
        ethans_war_party.board = [DragonspawnLieutenant(), DragonspawnLieutenant(), AlleyCat(), PackLeader()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)


if __name__ == '__main__':
    unittest.main()
