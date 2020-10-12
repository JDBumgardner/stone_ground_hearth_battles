import unittest
from typing import Type

from hearthstone.simulator.core.adaptations import AdaptBuffs
from hearthstone.simulator.core.card_graveyard import *
from hearthstone.simulator.core.card_pool import *

from hearthstone.simulator.core.combat import WarParty, fight_boards
from hearthstone.simulator.core.hero_pool import *
from hearthstone.simulator.core.player import Player
from hearthstone.simulator.core.randomizer import DefaultRandomizer
from hearthstone.simulator.core.tavern import Tavern


class CombatTests(unittest.TestCase):
    class TauntTestRandomizer(DefaultRandomizer):
        def select_attack_target(self, defenders: List[MonsterCard]) -> MonsterCard:
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
        def select_attack_target(self, defenders: List[MonsterCard]) -> MonsterCard:
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
        ethans_war_party.board = [IronhideRunt()]
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
        adams_war_party.board = [RighteousProtector(), RighteousProtector(), RighteousProtector(), RighteousProtector(),
                                 RighteousProtector(), RighteousProtector(), RighteousProtector()]
        ethans_war_party.board = [RighteousProtector(), RighteousProtector(), RighteousProtector(),
                                  RighteousProtector(), RighteousProtector(), RighteousProtector(), RatPack()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 37)
        self.assertEqual(ethan.health, 40)

    def test_arcane_cannon(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ArcaneCannon(), TwilightEmissary()]
        ethans_war_party.board = [DeckSwabbie(), DeckSwabbie(), DeckSwabbie(), DragonspawnLieutenant()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_arcane_cannon_scallywag(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        cat = AlleyCat()
        cat.taunt = True
        adams_war_party.board = [Scallywag(), ArcaneCannon(), VulgarHomunculus()]
        ethans_war_party.board = [BloodsailCannoneer(), MurlocTidehunter(), MurlocTidehunter(), MurlocTidehunter(), MurlocTidehunter(), cat]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

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
        def select_enemy_minion(self, enemy_minions: List[MonsterCard]) -> MonsterCard:
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
        ethans_war_party.board = [BloodsailCannoneer(), Rat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertTrue(isinstance(adams_war_party.board[1], Imp))

    def test_infested_wolf(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [InfestedWolf()]
        ethans_war_party.board = [FreedealingGambler(), RighteousProtector()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_monstrous_macaw(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MonstrousMacaw(), MechaRoo(), AlleyCat()]
        ethans_war_party.board = [TwilightEmissary(), StewardOfTime()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adams_war_party.board), 5)

    def test_piloted_shredder(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [PilotedShredder()]
        ethans_war_party.board = [FreedealingGambler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertNotEqual(ethan.health, 40)
        self.assertNotEqual(len(adams_war_party.board), 1)
        self.assertEqual(adams_war_party.board[1].mana_cost, 2)

    def test_soul_juggler(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [IronhideRunt()]
        ethans_war_party.board = [Imp(), SoulJuggler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_golden_soul_juggler(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [HarvestGolem(), Khadgar()]
        ethans_war_party.board = [RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 34)

    class PilotedShredderRandomizer(DefaultRandomizer):
        def select_summon_minion(self, card_types: List['Type']) -> 'Type':
            khadgar = [card_type for card_type in card_types if card_type == Khadgar]
            return khadgar[0]

    def test_khadgar_piloted_shredder(self):
        logging.basicConfig(level=logging.DEBUG)
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        piloted_shredder = PilotedShredder()
        piloted_shredder.golden_transformation([])
        adams_war_party.board = [piloted_shredder, Khadgar()]
        ethans_war_party.board = [MalGanis()]
        fight_boards(adams_war_party, ethans_war_party, self.PilotedShredderRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 18)

    def test_savannah_highmane(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SavannahHighmane()]
        ethans_war_party.board = [IronhideRunt()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 37)
        self.assertEqual(len(adams_war_party.board), 3)

    def test_security_rover(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SecurityRover()]
        ethans_war_party.board = [RabidSaurolisk(), RabidSaurolisk(), RabidSaurolisk()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 38)
        self.assertEqual(len(adams_war_party.board), 3)

    def test_ripsnarl_captain(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [FreedealingGambler(), RipsnarlCaptain()]
        ethans_war_party.board = [IronhideRunt()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 35)

    def test_southsea_captain(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [FreedealingGambler(), SouthseaCaptain()]
        ethans_war_party.board = [TwilightEmissary()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adams_war_party.board[0].attack, 4)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 37)

    def test_sneeds_old_shredder(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [RighteousProtector(), DrakonidEnforcer()]
        houndmaster = Houndmaster()
        houndmaster.golden_transformation([])
        ethans_war_party.board = [houndmaster]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_bronze_warden(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ReplicatingMenace()]
        ethans_war_party.board = [RatPack(), Rat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adams_war_party.board), 4)

    def test_junkbot(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [KaboomBot(), Junkbot()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_voidlord(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [DeckSwabbie(), DreadAdmiralEliza()]
        ironhide_runt = IronhideRunt()
        ironhide_runt.golden_transformation([])
        ethans_war_party.board = [ironhide_runt]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_goldrinn_the_great_wolf(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [NadinaTheRed(), StewardOfTime()]
        ethans_war_party.board = [KalecgosArcaneAspect()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_the_tide_razor(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [Maexxna()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_herald_of_flame(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [HeraldOfFlame(), Rat(), Rat()]
        ethans_war_party.board = [Robosaur(), UnstableGhoul()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_herald_of_flame_chain(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [HeraldOfFlame(), Rat(), Rat(), Rat(), Rat(), Rat(), Rat()]
        ethans_war_party.board = [MicroMachine(), MicroMachine(), MicroMachine(), MicroMachine(), MicroMachine(),
                                  UnstableGhoul()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 35)

    def test_herald_of_flame_defender(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [IronhideDirehorn(), Rat()]
        ethans_war_party.board = [NadinaTheRed()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 37)

    def test_ironhide_direhorn_defender(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [NatPagleExtremeAngler()]
        ethans_war_party.board = [IronhideRunt(), Rat()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(adam.hand_size(), 0)

    def test_mal_ganis(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [OldMurkeye()]
        ethans_war_party.board = [KingBagurgle()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_baron_rivendare(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [KindlyGrandmother(), BaronRivendare()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_macaw_goldrinn(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MonstrousMacaw(), GoldrinnTheGreatWolf()]
        ethans_war_party.board = [FreedealingGambler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertFalse(adams_war_party.board[0].dead)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 30)
        self.assertEqual(adams_war_party.board[0].attack, 9)
        self.assertEqual(adams_war_party.board[1].attack, 9)
        self.assertEqual(adams_war_party.board[1].health, 9)

    def test_macaw_baron(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MonstrousMacaw(), GoldrinnTheGreatWolf(), BaronRivendare()]
        ethans_war_party.board = [FreedealingGambler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertFalse(adams_war_party.board[0].dead)
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 25)
        self.assertEqual(adams_war_party.board[0].attack, 14)
        self.assertEqual(adams_war_party.board[1].attack, 14)
        self.assertEqual(adams_war_party.board[1].health, 14)

    def test_zero_attack(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [YoHoOgre(), RipsnarlCaptain()]
        ethans_war_party.board = [DeckSwabbie(), TwilightEmissary(), Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_waxrider_togwaggle(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TwilightEmissary(), WaxriderTogwaggle()]
        ethans_war_party.board = [TwilightEmissary(), Houndmaster()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_herald_togwaggle(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SavannahHighmane(), MamaBear(), VulgarHomunculus()]
        ethans_war_party.board = [BloodsailCannoneer(), NadinaTheRed(), NadinaTheRed(), NadinaTheRed(), NadinaTheRed()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_micro_mummy(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [MicroMummy()]
        ethans_war_party.board = [BloodsailCannoneer()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_amalgadon(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SecurityRover(), KangorsApprentice()]
        ethans_war_party.board = [Maexxna()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 35)

    def test_zapp_slywick(self): # TODO I don't think this actually tests everything about this card
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [ZappSlywick()]
        ethans_war_party.board = [RighteousProtector(), MonstrousMacaw(), MechanoEgg()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 38)
        self.assertEqual(ethan.health, 40)

    def test_foe_reaper_4000(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [FoeReaper4000()]
        ethans_war_party.board = [AlleyCat(), MamaBear(), DragonspawnLieutenant(), MamaBear()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 33)

    def test_macaw_multiple_deathrattles(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        zapp = ZappSlywick()
        zapp.golden_transformation([])
        adams_war_party.board = [zapp, VulgarHomunculus()]
        ethans_war_party.board = [BloodsailCannoneer(), DeckSwabbie(), RabidSaurolisk(), CapnHoggarr(), NatPagleExtremeAngler()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_seabreaker_goliath(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [SeabreakerGoliath(), DeckSwabbie(), DeckSwabbie(), VulgarHomunculus()]
        ethans_war_party.board = [BloodsailCannoneer(), CapnHoggarr(), CapnHoggarr(), TwilightEmissary(),
                                  TwilightEmissary()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_siege_breaker(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [VulgarHomunculus(), Siegebreaker()]
        ethans_war_party.board = [Robosaur()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_the_beast(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [TheBeast(), RabidSaurolisk()]
        ethans_war_party.board = [NadinaTheRed()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_khadgar_the_beast(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        twilight_emissary = TwilightEmissary()
        twilight_emissary.golden_transformation([])
        khadgar = Khadgar()
        khadgar.attack += 1
        adams_war_party.board = [TheBeast(), khadgar, RabidSaurolisk()]
        ethans_war_party.board = [twilight_emissary]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)
        self.assertEqual(len(adams_war_party.board), 3)

    def test_crackling_cyclone(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
        adams_war_party = WarParty(adam)
        ethans_war_party = WarParty(ethan)
        adams_war_party.board = [CracklingCyclone(), VulgarHomunculus()]
        ethans_war_party.board = [BloodsailCannoneer(), VulgarHomunculus(), VulgarHomunculus()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)

    def test_deadly_spore(self):
        adam = Player.new_player_with_hero(None, "Adam")
        ethan = Player.new_player_with_hero(None, "Ethan")
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
        ethans_war_party.board = [Robosaur()]
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
        ethans_war_party.board = [BloodsailCannoneer(), CapnHoggarr(), LieutenantGarr(), CapnHoggarr()]
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
        adams_war_party.board = [wildfire_elemental, VulgarHomunculus()]
        ethans_war_party.board = [BloodsailCannoneer(), CapnHoggarr(), LieutenantGarr(), CapnHoggarr()]
        fight_boards(adams_war_party, ethans_war_party, DefaultRandomizer())
        self.assertEqual(adam.health, 40)
        self.assertEqual(ethan.health, 40)


if __name__ == '__main__':
    unittest.main()
