import itertools
import typing
from collections import defaultdict
from typing import Optional, List, Callable, Type

from hearthstone import events
from hearthstone.cards import MonsterCard
from hearthstone.events import BuyPhaseContext, CardEvent
from hearthstone.hero import EmptyHero
from hearthstone.monster_types import MONSTER_TYPES
from hearthstone.special_cards import TripleRewardCard

if typing.TYPE_CHECKING:
    from hearthstone.tavern import Tavern
    from hearthstone.hero import Hero
    from hearthstone.randomizer import Randomizer


class BuyPhaseEvent:
    pass


StoreIndex = typing.NewType("StoreIndex", int)
HandIndex = typing.NewType("HandIndex", int)
BoardIndex = typing.NewType("BoardIndex", int)


class Player:
    def __init__(self, tavern: 'Tavern', name: str, hero_options: List['Hero']):
        self.name = name
        self.tavern = tavern
        self.hero = None
        self.hero_options = hero_options
        self.health = None
        self.tavern_tier = 1
        self._coins = 0
        self.triple_rewards = []
        self.discover_queue: List[List['MonsterCard']] = []
        self.maximum_board_size = 7
        self.maximum_hand_size = 10
        self.refresh_store_cost = 1
        self._tavern_upgrade_costs = (0, 5, 7, 8, 9, 10)
        self.tavern_upgrade_cost = 5
        self.hand: List[MonsterCard] = []
        self.in_play: List[MonsterCard] = []
        self.store: List[MonsterCard] = []
        self.frozen = False
        self.counted_cards = defaultdict(lambda: 0)
        self.minion_cost = 3
        self.gold_coins = 0
        self.bananas = 0
        self.purchased_minions: List['Type'] = []
        self.last_opponent_warband: List['MonsterCard'] = []
        self.recruitment_maps = []
        self.dead = False

    @property
    def coins(self):
        return self._coins

    @coins.setter
    def coins(self, coins):
        self._coins = min(coins, 10)

    @staticmethod
    def new_player_with_hero(tavern: Optional['Tavern'], name: str, hero: Optional['Hero'] = None) -> 'Player':
        if hero is None:
            hero = EmptyHero()
        player = Player(tavern, name, [hero])
        player.choose_hero(hero)
        return player

    @property
    def coin_income_rate(self):
        return min(self.tavern.turn_count + 3, 10)

    def buying_step(self):
        self.reset_purchased_minions_list()
        self.apply_turn_start_income()
        self.draw()
        self.hero.on_buy_step()
        self.broadcast_buy_phase_event(events.BuyStartEvent())

    def reset_purchased_minions_list(self):
        self.purchased_minions = []

    def apply_turn_start_income(self):
        self.coins = self.coin_income_rate

    def decrease_tavern_upgrade_cost(self):
        self.tavern_upgrade_cost = max(0, self.tavern_upgrade_cost - 1)

    def upgrade_tavern(self):
        assert self.valid_upgrade_tavern()
        self.coins -= self.tavern_upgrade_cost
        self.tavern_tier += 1
        if self.tavern_tier < self.max_tier():
            self.tavern_upgrade_cost = self._tavern_upgrade_costs[self.tavern_tier]
        self.broadcast_buy_phase_event(events.TavernUpgradeEvent())

    def valid_upgrade_tavern(self) -> bool:
        if self.tavern_tier >= self.max_tier():
            return False
        if self.coins < self.tavern_upgrade_cost:
            return False
        return True

    def summon_from_hand(self, index: HandIndex, targets: Optional[List[BoardIndex]] = None):
        if targets is None:
            targets = []
        assert self.valid_summon_from_hand(index, targets)
        card = self.hand.pop(index)
        self.in_play.append(card)
        if card.golden:
            self.triple_rewards.append(TripleRewardCard(min(self.tavern_tier + 1, 6)))
        target_cards = [self.in_play[target] for target in targets]
        self.broadcast_buy_phase_event(events.SummonBuyEvent(card, target_cards))

    def valid_summon_from_hand(self, index: HandIndex, targets: Optional[List[BoardIndex]] = None) -> bool:
        if targets is None:
            targets = []
        #  TODO: Jack num_battlecry_targets should only accept 0,1,2
        for target in targets:
            if not self.valid_board_index(target):
                return False
        if not self.valid_hand_index(index):
            return False
        card = self.hand[index]
        if not self.room_on_board():
            return False
        if card.battlecry:
            valid_targets = [target_index for target_index, target_card in enumerate(self.in_play) if card.valid_battlecry_target(target_card)]
            possible_num_targets = [num_targets for num_targets in card.num_battlecry_targets if num_targets <= len(valid_targets)]
            if not possible_num_targets:
                possible_num_targets = [len(valid_targets)]
            if len(targets) not in possible_num_targets:
                return False
            if len(set(targets)) != len(targets):
                return False
            for target in targets:
                if target not in valid_targets:
                    return False
        if card.magnetic:
            if len(targets) > 1:
                return False
            if len(targets) == 1 and not self.in_play[targets[0]].check_type(MONSTER_TYPES.MECH):
                return False
        return True

    def play_triple_rewards(self):
        if not self.triple_rewards:
            return
        discover_tier = self.triple_rewards.pop(-1).level
        self.draw_discover(lambda card: card.tier == discover_tier)

    def valid_triple_rewards(self) -> bool:
        return bool(self.triple_rewards)

    def draw_discover(self, predicate: Callable[['MonsterCard'], bool]): #TODO: Jarett help make discoverables unique are cards with more copies in the deck more likely to be discovered?
        discoverables = [card for card in self.tavern.deck.unique_cards() if predicate(card)]
        discovered_cards = []
        for _ in range(3):
            discovered_cards.append(self.tavern.randomizer.select_discover_card(discoverables))
            discoverables.remove(discovered_cards[-1])
            self.tavern.deck.remove_card(discovered_cards[-1])
        self.discover_queue.append(discovered_cards)

    def select_discover(self, card: 'MonsterCard'):
        assert (card in self.discover_queue[0])
        assert (isinstance(card, MonsterCard))
        self.discover_queue[0].remove(card)
        self.gain_card(card)
        self.tavern.deck.return_cards(itertools.chain.from_iterable([card.dissolve() for card in self.discover_queue[0]]))
        self.discover_queue.pop(0)

    def summon_from_void(self, monster: MonsterCard):
        if self.room_on_board():
            self.in_play.append(monster)
            self.check_golden(type(monster))
            self.broadcast_buy_phase_event(events.SummonBuyEvent(monster))

    def room_on_board(self):
        return len(self.in_play) < self.maximum_board_size

    def draw(self):
        if self.frozen:
            self.frozen = False
        else:
            self.return_cards()
        number_of_cards = 3 + self.tavern_tier // 2 - len(self.store)
        self.store.extend(self.tavern.deck.draw(self, number_of_cards))

    def purchase(self, index: StoreIndex):
        # check if the index is valid
        assert self.valid_purchase(index)
        card = self.store.pop(index)
        self.coins -= self.minion_cost
        self.hand.append(card)
        event = events.BuyEvent(card)
        self.broadcast_buy_phase_event(event)
        self.check_golden(type(card))
        self.purchased_minions.append(type(card))

    def valid_purchase(self, index: 'StoreIndex') -> bool:
        if not self.valid_store_index(index):
            return False
        if self.coins < self.minion_cost:
            return False
        if not self.room_in_hand():
            return False
        return True

    def check_golden(self, check_card: Type[MonsterCard]):
        cards = [card for card in self.in_play + self.hand if isinstance(card, check_card) and not card.golden]
        assert len(cards) <= 3, f"fnord{cards}"
        if len(cards) == 3:
            for card in cards:
                if card in self.in_play:
                    self.in_play.remove(card)
                if card in self.hand:
                    self.hand.remove(card)
            golden_card = check_card()
            golden_card.golden_transformation(cards)
            self.hand.append(golden_card)

    def reroll_store(self):
        assert self.valid_reroll()
        self.coins -= self.refresh_store_cost
        self.draw()
        self.broadcast_buy_phase_event(events.RefreshStoreEvent())

    def valid_reroll(self) -> bool:
        return self.coins >= self.refresh_store_cost

    def return_cards(self):
        self.tavern.deck.return_cards(itertools.chain.from_iterable([card.dissolve() for card in self.store]))
        self.store = []

    def freeze(self):
        self.frozen = True

    def sell_minion(self, index: BoardIndex):
        assert self.valid_sell_minion(index)
        card = self.in_play[index]
        self.broadcast_buy_phase_event(events.SellEvent(card))
        self.in_play.remove(card)
        self.coins += card.redeem_rate
        returned_cards = card.dissolve()
        self.tavern.deck.return_cards(returned_cards)

    def valid_sell_minion(self, index: 'BoardIndex') -> bool:
        return self.valid_board_index(index)

    def hero_power(self, board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        self.hero.hero_power(BuyPhaseContext(self, self.tavern.randomizer), board_index, store_index)

    def valid_hero_power(self, board_target: Optional['BoardIndex'] = None, store_target: Optional['StoreIndex'] = None) -> bool:
        return self.hero.hero_power_valid(BuyPhaseContext(self, self.tavern.randomizer), board_target, store_target)

    def broadcast_buy_phase_event(self, event: CardEvent, randomizer: Optional['Randomizer'] = None):
        self.hero.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.in_play.copy():
            card.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.hand.copy():
            card.handle_event_in_hand(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))

    def hand_size(self):
        return len(self.hand) + len(self.triple_rewards) + self.gold_coins + self.bananas + len(self.recruitment_maps)

    def room_in_hand(self):
        return self.hand_size() < self.maximum_hand_size

    def max_tier(self):
        return len(self._tavern_upgrade_costs)

    def choose_hero(self, hero: 'Hero'):
        assert(self.valid_choose_hero(hero))
        self.hero = hero
        self.hero_options = []
        self.health = self.hero.starting_health()
        self.minion_cost = self.hero.minion_cost()
        self.refresh_store_cost = self.hero.refresh_cost()
        self._tavern_upgrade_costs = self.hero.tavern_upgrade_costs()
        self.tavern_upgrade_cost = self.hero.tavern_upgrade_costs()[1]

    def valid_choose_hero(self, hero: 'Hero'):
        return self.hero is None and hero in self.hero_options

    def take_damage(self, damage: int):
        if not any(card.give_immunity for card in self.in_play):
            self.health -= damage
            self.broadcast_buy_phase_event(events.PlayerDamagedEvent())

    def redeem_gold_coin(self):
        if self.gold_coins >= 1:
            self.gold_coins -= 1
            self.coins += 1

    def gain_card(self, card: 'MonsterCard'):
        self.hand.append(card)
        self.check_golden(type(card))

    def valid_board_index(self, index: 'BoardIndex') -> bool:
        return 0 <= index < len(self.in_play)

    def valid_hand_index(self, index: 'HandIndex') -> bool:
        return 0 <= index < len(self.hand)

    def valid_store_index(self, index: 'StoreIndex') -> bool:
        return 0 <= index < len(self.store)

    def plus_coins(self, amt: int):
        self.coins = min(self.coins + amt, 10)

    def use_banana(self, board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        assert self.valid_use_banana(board_index, store_index)
        self.bananas -= 1
        if board_index is not None:
            self.in_play[board_index].attack += 1
            self.in_play[board_index].health += 1
        if store_index is not None:
            self.store[store_index].attack += 1
            self.store[store_index].health += 1

    def valid_use_banana(self, board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        if board_index == store_index:
            return False
        if board_index is not None and not self.in_play:
            return False
        if store_index is not None and not self.store:
            return False
        return True

    def play_recruitment_map(self):
        assert self.valid_play_recruitment_map()
        recruitment_map = self.recruitment_maps.pop()
        self.coins -= recruitment_map.cost
        discover_tier = recruitment_map.level
        self.draw_discover(lambda card: card.tier == discover_tier)

    def valid_play_recruitment_map(self):
        return bool(self.recruitment_maps) and self.coins >= self.recruitment_maps[-1].cost

    def resolve_death(self):
        assert not self.dead and self.health <= 0
        self.dead = True
        self.broadcast_self_death_event(events.PlayerDeadEvent(self))
        self.tavern.deck.return_cards(itertools.chain.from_iterable([card.dissolve() for card in self.in_play]))

    def broadcast_self_death_event(self, event: 'CardEvent', randomizer: Optional['Randomizer'] = None):  # Perhaps there is a better implementation than this
        for player in self.tavern.players.values():
            player.hero.handle_event(event, BuyPhaseContext(player, randomizer or self.tavern.randomizer))

