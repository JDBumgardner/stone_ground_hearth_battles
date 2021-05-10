import copy
import logging
import typing
from inspect import isclass
from typing import Union

from hearthstone.simulator.core.events import CardEvent, BuyPhaseContext, CombatPhaseContext, EVENTS

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.hero import Hero

logger = logging.getLogger(__name__)


class Secret:
    def __repr__(self):
        return "{" + f"{type(self).__name__}" + "}"

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        pass


class BaseSecret:
    class IceBlock(Secret):
        def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
            if event.event is EVENTS.END_COMBAT and context.owner.health <= 0:
                logger.debug(f'{self} triggers')
                context.owner.hero.secrets.remove(self)
                context.owner.health += event.damage_taken
                context.owner.hero.give_immunity = True

    class SplittingImage(Secret):
        def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
            if event.event is EVENTS.IS_ATTACKED and event.card in context.friendly_war_party.board and context.friendly_war_party.room_on_board():
                logger.debug(f'{self} triggers')
                context.friendly_war_party.owner.hero.secrets.remove(self)
                summon_index = context.friendly_war_party.get_index(event.card)
                for i in range(context.summon_minion_multiplier()):
                    context.friendly_war_party.summon_in_combat(copy.deepcopy(event.card), context, summon_index + 1 + i)

    class AutodefenseMatrix(Secret):
        def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
            if event.event is EVENTS.IS_ATTACKED and event.card in context.friendly_war_party.board and not event.card.divine_shield:
                logger.debug(f'{self} triggers')
                context.friendly_war_party.owner.hero.secrets.remove(self)
                event.card.divine_shield = True

    class VenomstrikeTrap(Secret):
        def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
            if event.event is EVENTS.IS_ATTACKED and event.card in context.friendly_war_party.board and context.friendly_war_party.room_on_board():

                from hearthstone.simulator.core.card_pool import EmperorCobra  # is this the best way to avoid circular imports?

                logger.debug(f'{self} triggers')
                context.friendly_war_party.owner.hero.secrets.remove(self)
                for _ in range(context.summon_minion_multiplier()):
                    cobra = EmperorCobra()
                    context.friendly_war_party.summon_in_combat(cobra, context)

    class Redemption(Secret):
        def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
            if event.event is EVENTS.DIES and event.card in context.friendly_war_party.board:
                logger.debug(f'{self} triggers')
                context.friendly_war_party.owner.hero.secrets.remove(self)
                summon_index = context.friendly_war_party.get_index(event.card)
                for i in range(context.summon_minion_multiplier()):
                    new_copy = event.card.unbuffed_copy()
                    new_copy.health = 1
                    context.friendly_war_party.summon_in_combat(new_copy, context, summon_index + 1 + i)

    class Avenge(Secret):
        def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
            if event.event is EVENTS.DIES and event.card in context.friendly_war_party.board and context.friendly_war_party.live_minions():
                logger.debug(f'{self} triggers')
                context.friendly_war_party.owner.hero.secrets.remove(self)
                random_friend = context.randomizer.select_friendly_minion(context.friendly_war_party.live_minions())
                random_friend.attack += 3
                random_friend.health += 2

    class SnakeTrap(Secret):
        def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
            if event.event is EVENTS.IS_ATTACKED and event.card in context.friendly_war_party.board and context.friendly_war_party.room_on_board():

                from hearthstone.simulator.core.card_pool import Snake  # is this the best way to avoid circular imports?

                logger.debug(f'{self} triggers')
                context.friendly_war_party.owner.hero.secrets.remove(self)
                for _ in range(3 * context.summon_minion_multiplier()):
                    snake = Snake()
                    context.friendly_war_party.summon_in_combat(snake, context)

    class CompetitiveSpirit(Secret):
        def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
            if event.event is EVENTS.BUY_START:
                logger.debug(f'{self} triggers')
                context.owner.hero.secrets.remove(self)
                for card in context.owner.in_play:
                    card.attack += 1
                    card.health += 1


def remaining_secrets(hero: 'Hero'):
    return [secret for secret in ALL_SECRETS if secret not in hero.secrets]


ALL_SECRETS = [secret for secret in BaseSecret.__dict__.values() if isclass(secret)]
