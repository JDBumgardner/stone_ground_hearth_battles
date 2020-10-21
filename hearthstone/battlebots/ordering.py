from hearthstone.simulator.agent import RearrangeCardsAction
from hearthstone.simulator.core.card_pool import *
from hearthstone.simulator.core.tavern import Player


def rate_position(card: 'MonsterCard') -> float:
    if type(card) is MonstrousMacaw:
        return 0.0
    if type(card) is UnstableGhoul or type(card) is SpawnOfNzoth:
        return 1.0
    if type(card) is SelflessHero or type(card) is GlyphGuardian or type(card) is DeflectOBot:
        return 2.0
    if type(card) is OldMurkeye:
        return 3.0
    if type(card) is InfestedWolf or type(card) is SavannahHighmane or type(card) is SecurityRover:
        return 4.5
    if type(card) is DragonspawnLieutenant or type(card) is RighteousProtector or type(card) is Imprisoner or type(
            card) is ImpGangBoss or type(card) is TwilightEmissary:
        return 5.0
    if type(card) is ScavengingHyena or type(card) is RatPack:
        return 6.0
    if type(card) is PackLeader or type(card) is MurlocWarleader or type(card) is Khadgar or type(
            card) is SouthseaCaptain:
        return 6.5
    if type(card) is MamaBear or type(card) is SoulJuggler or type(card) is RipsnarlCaptain:
        return 7.0

    return 4.0


def naive_rearrange_cards(player: 'Player') -> RearrangeCardsAction:
    in_play = list(enumerate(player.in_play))
    in_play.sort(key=lambda pair: rate_position(pair[1]))
    return RearrangeCardsAction([pair[0] for pair in in_play])
