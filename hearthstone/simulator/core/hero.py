import copy
import typing
from typing import Union, Tuple, Optional, List, Any

from hearthstone.simulator.core.card_pool import EmperorCobra, Snake
from hearthstone.simulator.core.cards import CardLocation
from hearthstone.simulator.core.combat import logger
from hearthstone.simulator.core.events import BuyPhaseContext, CombatPhaseContext, CardEvent, EVENTS
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.secrets import SECRETS

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.player import BoardIndex, StoreIndex, DiscoverIndex


class Hero:
    base_power_cost: Optional[int] = None  # default value is for heroes with passive hero powers
    hero_power_used = False
    can_use_power = True
    power_target_location: Optional[List['CardLocation']] = None
    multiple_power_uses_per_turn = False
    pool: 'MONSTER_TYPES' = MONSTER_TYPES.ALL

    def __init__(self):
        self.power_cost = self.base_power_cost
        self.discover_queue: List[List[Any]] = []
        self.secrets = []
        self.give_immunity = False

    def __repr__(self):
        return str(type(self).__name__)

    def starting_health(self) -> int:
        return 40

    def minion_cost(self) -> int:
        return 3

    def refresh_cost(self) -> int:
        return 1

    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return 0, 5, 7, 8, 9, 10

    def occupied_store_slots(self) -> int:
        return 0

    def occupied_hand_slots(self) -> int:
        return 0

    def handle_event(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_END:
            self.give_immunity = False
        if len(self.secrets) > 0:
            self.handle_secrets(event, context)
        self.handle_event_powers(event, context)

    def handle_event_powers(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        pass

    def hero_power(self, context: BuyPhaseContext, board_index: Optional['BoardIndex'] = None,
                   store_index: Optional['StoreIndex'] = None):
        assert self.hero_power_valid(context, board_index, store_index)
        context.owner.coins -= self.power_cost
        self.hero_power_used = True
        self.hero_power_impl(context, board_index, store_index)

    def hero_power_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                        store_index: Optional['StoreIndex'] = None):
        pass

    def hero_power_valid(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                         store_index: Optional['StoreIndex'] = None):
        if self.power_cost is None:
            return False
        if context.owner.coins < self.power_cost:
            return False
        if not self.multiple_power_uses_per_turn:
            if self.hero_power_used:
                return False
        if not self.can_use_power:
            return False
        if self.power_target_location is None and (board_index is not None or store_index is not None):
            return False
        if self.power_target_location is not None:
            if board_index is None and store_index is None:
                return False
            if board_index is not None:
                if CardLocation.BOARD not in self.power_target_location or not context.owner.valid_board_index(
                        board_index):
                    return False
            if store_index is not None:
                if CardLocation.STORE not in self.power_target_location or not context.owner.valid_store_index(
                        store_index):
                    return False
        if not self.hero_power_valid_impl(context, board_index, store_index):
            return False
        return True

    def hero_power_valid_impl(self, context: 'BuyPhaseContext', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        return True

    def on_buy_step(self):
        self.hero_power_used = False

    def battlecry_multiplier(self) -> int:
        return 1

    def select_discover(self, discover_index: 'DiscoverIndex', context: 'BuyPhaseContext'):
        pass

    def valid_select_discover(self, discover_index: 'DiscoverIndex') -> bool:
        return bool(self.discover_queue) and discover_index in range(len(self.discover_queue[0]))

    def hero_info(self) -> Optional[str]:
        return None

    def handle_secrets(self, event: 'CardEvent', context: Union['BuyPhaseContext', 'CombatPhaseContext']):
        if event.event is EVENTS.BUY_START and SECRETS.COMPETETIVE_SPIRIT in self.secrets:
            logger.debug(f'{SECRETS.COMPETETIVE_SPIRIT} triggers')
            self.secrets.remove(SECRETS.COMPETETIVE_SPIRIT)
            for card in context.owner.in_play:
                card.attack += 1
                card.health += 1
        if event.event is EVENTS.END_COMBAT and context.owner.health <= 0 and SECRETS.ICE_BLOCK in self.secrets:
            logger.debug(f'{SECRETS.ICE_BLOCK} triggers')
            self.secrets.remove(SECRETS.ICE_BLOCK)
            context.owner.health += event.damage_taken
            self.give_immunity = True
        if event.event is EVENTS.IS_ATTACKED and event.card in context.friendly_war_party.board:
            if context.friendly_war_party.room_on_board():
                if SECRETS.SPLITTING_IMAGE in self.secrets and context.friendly_war_party.room_on_board():
                    logger.debug(f'{SECRETS.SPLITTING_IMAGE} triggers')
                    self.secrets.remove(SECRETS.SPLITTING_IMAGE)
                    summon_index = context.friendly_war_party.get_index(event.card)
                    for i in range(context.summon_minion_multiplier()):
                        context.friendly_war_party.summon_in_combat(copy.deepcopy(event.card), context,
                                                                    summon_index + 1 + i)
                if SECRETS.VENOMSTRIKE_TRAP in self.secrets and context.friendly_war_party.room_on_board():
                    logger.debug(f'{SECRETS.VENOMSTRIKE_TRAP} triggers')
                    self.secrets.remove(SECRETS.VENOMSTRIKE_TRAP)
                    for _ in range(context.summon_minion_multiplier()):
                        cobra = EmperorCobra()
                        context.friendly_war_party.summon_in_combat(cobra, context)
                if SECRETS.SNAKE_TRAP in self.secrets and context.friendly_war_party.room_on_board():
                    self.secrets.remove(SECRETS.SNAKE_TRAP)
                    logger.debug(f'{SECRETS.SNAKE_TRAP} triggers')
                    for _ in range(3 * context.summon_minion_multiplier()):
                        snake = Snake()
                        context.friendly_war_party.summon_in_combat(snake, context)
            if SECRETS.AUTODEFENSE_MATRIX in self.secrets and not event.card.divine_shield:
                logger.debug(f'{SECRETS.AUTODEFENSE_MATRIX} triggers')
                self.secrets.remove(SECRETS.AUTODEFENSE_MATRIX)
                event.card.divine_shield = True

        if event.event is EVENTS.DIES and event.card in context.friendly_war_party.board:
            if SECRETS.REDEMPTION in self.secrets:
                logger.debug(f'{SECRETS.REDEMPTION} triggers')
                self.secrets.remove(SECRETS.REDEMPTION)
                summon_index = context.friendly_war_party.get_index(event.card)
                for i in range(context.summon_minion_multiplier()):
                    new_copy = event.card.unbuffed_copy()
                    new_copy.health = 1
                    context.friendly_war_party.summon_in_combat(new_copy, context, summon_index + 1 + i)

            if SECRETS.AVENGE in self.secrets and context.friendly_war_party.live_minions():
                logger.debug(f'{SECRETS.AVENGE} triggers')
                self.secrets.remove(SECRETS.AVENGE)
                random_friend = context.randomizer.select_friendly_minion(context.friendly_war_party.live_minions())
                random_friend.attack += 3
                random_friend.health += 2


class EmptyHero(Hero):
    pass
