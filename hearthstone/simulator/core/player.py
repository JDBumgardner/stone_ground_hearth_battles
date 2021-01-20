import itertools
import typing
from typing import Optional, List, Callable, Type, Tuple

from frozenlist.frozen_list import FrozenList
from hearthstone.simulator.core import events
from hearthstone.simulator.core.cards import MonsterCard
from hearthstone.simulator.core.events import BuyPhaseContext, CardEvent
from hearthstone.simulator.core.hero import EmptyHero
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.triple_reward_card import TripleRewardCard

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.tavern import Tavern
    from hearthstone.simulator.core.hero import Hero
    from hearthstone.simulator.core.randomizer import Randomizer


class BuyPhaseEvent:
    pass


StoreIndex = typing.NewType("StoreIndex", int)
HandIndex = typing.NewType("HandIndex", int)
BoardIndex = typing.NewType("BoardIndex", int)
DiscoverIndex = typing.NewType("DiscoverIndex", int)
HeroChoiceIndex = typing.NewType("HeroChoiceIndex", int)


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
        self.maximum_store_size = 7
        self.refresh_store_cost = 1
        self._tavern_upgrade_costs = (0, 5, 7, 8, 9, 10)
        self._tavern_upgrade_cost = 5
        self._hand: List[MonsterCard] = []
        self._in_play: List[MonsterCard] = []
        self._store: List[MonsterCard] = []
        self.minion_cost = 3
        self.gold_coins = 0
        self.bananas = 0  # tracks total number of bananas (big and small)
        self.big_bananas = 0  # tracks how many bananas are big
        self.purchased_minions: List['Type'] = []
        self.played_minions: List['Type'] = []
        self.last_opponent_warband: List['MonsterCard'] = []
        self.dead = False
        self.nomi_bonus = 0
        self.free_refreshes = 0

    def __repr__(self):
        return f"{self.hero} ({self.name})"

    @property
    def coins(self):
        return self._coins

    @coins.setter
    def coins(self, coins):
        self._coins = max(0, min(coins, 10))

    @property
    def tavern_upgrade_cost(self):
        return self._tavern_upgrade_cost

    @tavern_upgrade_cost.setter
    def tavern_upgrade_cost(self, tavern_upgrade_cost):
        self._tavern_upgrade_cost = max(0, tavern_upgrade_cost)

    @property
    def in_play(self):
        return FrozenList(self._in_play)

    @property
    def hand(self):
        return FrozenList(self._hand)

    @property
    def store(self):
        return FrozenList(self._store)

    @staticmethod
    def new_player_with_hero(tavern: Optional['Tavern'], name: str, hero: Optional['Hero'] = None) -> 'Player':
        if hero is None:
            hero = EmptyHero()
        player = Player(tavern, name, [hero])
        player.choose_hero(HeroChoiceIndex(0))
        return player

    @property
    def coin_income_rate(self):
        return min(self.tavern.turn_count + 3, 10)

    def buying_step(self):
        self.reset_purchased_minions_list()
        self.reset_played_minions_list()
        self.apply_turn_start_income()
        self.draw(unfreeze=False)
        self.hero.on_buy_step()
        self.broadcast_buy_phase_event(events.BuyStartEvent())

    def reset_purchased_minions_list(self):
        self.purchased_minions = []

    def reset_played_minions_list(self):
        self.played_minions = []

    def apply_turn_start_income(self):
        self.coins = self.coin_income_rate

    def decrease_tavern_upgrade_cost(self):
        self.tavern_upgrade_cost -= 1

    def upgrade_tavern(self):
        assert self.valid_upgrade_tavern()
        self.coins -= self.tavern_upgrade_cost
        self.tavern_tier += 1
        if self.tavern_tier < self.max_tier():
            self.tavern_upgrade_cost = self._tavern_upgrade_costs[self.tavern_tier]
        self.broadcast_buy_phase_event(events.TavernUpgradeEvent())

    def valid_upgrade_tavern(self) -> bool:
        if self.dead:
            return False
        if self.discover_queue:
            return False
        if self.tavern_tier >= self.max_tier():
            return False
        if self.coins < self.tavern_upgrade_cost:
            return False
        return True

    def summon_from_hand(self, index: HandIndex, targets: Optional[List[BoardIndex]] = None):
        if targets is None:
            targets = []
        assert self.valid_summon_from_hand(index, targets)
        card = self._hand.pop(index)
        self.gain_board_card(card)
        if card.golden:
            self.triple_rewards.append(TripleRewardCard(min(self.tavern_tier + 1, 6)))
        target_cards = [self.in_play[target] for target in targets]
        self.broadcast_buy_phase_event(events.SummonBuyEvent(card, target_cards))
        self.played_minions.append(type(card))

    def valid_summon_from_hand(self, index: HandIndex, targets: Optional[List[BoardIndex]] = None) -> bool:
        if not self.room_to_summon(index):
            return False

        card = self.hand[index]
        if targets is None:
            targets = []
        #  TODO: Jack num_battlecry_targets should only accept 0,1,2
        for target in targets:
            if not self.valid_board_index(target):
                return False
        if card.battlecry:
            valid_targets = [target_index for target_index, target_card in enumerate(self.in_play) if
                             card.valid_battlecry_target(target_card)]
            possible_num_targets = [num_targets for num_targets in card.num_battlecry_targets if
                                    num_targets <= len(valid_targets)]
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

    def room_to_summon(self, index: HandIndex):
        if self.dead:
            return False
        if self.discover_queue:
            return False
        if not self.valid_hand_index(index):
            return False
        if not self.room_on_board():
            return False
        return True

    def play_triple_rewards(self):
        if not self.triple_rewards:
            return
        discover_tier = self.triple_rewards.pop(-1).level
        self.draw_discover(lambda card: card.tier == discover_tier)

    def valid_triple_rewards(self) -> bool:
        if self.dead:
            return False
        if self.discover_queue:
            return False
        return bool(self.triple_rewards)

    def draw_discover(self, predicate: Callable[[
                                                    'MonsterCard'], bool]):  # TODO: Jarett help make discoverables unique are cards with more copies in the deck more likely to be discovered?
        discoverables = [card for card in self.tavern.deck.all_cards() if predicate(
            card)]  # Jeremy says: Hmm, we can run out of unique cards.  Changed to be all cards for now.
        discovered_cards = []
        for _ in range(3):
            discovered_cards.append(self.tavern.randomizer.select_discover_card(discoverables))
            discoverables.remove(discovered_cards[-1])
            self.tavern.deck.remove_card(discovered_cards[-1])
        self.discover_queue.append(discovered_cards)

    def select_discover(self, card_index: 'DiscoverIndex'):
        assert self.valid_select_discover(card_index)
        card = self.discover_queue[0].pop(card_index)
        card.token = False  # for Bigglesworth (there is no other scenario where a token will be a discover option)
        self.gain_hand_card(card)
        self.tavern.deck.return_cards(
            itertools.chain.from_iterable([card.dissolve() for card in self.discover_queue[0]]))
        self.discover_queue.pop(0)

    def valid_select_discover(self, card_index: 'DiscoverIndex'):
        return self.discover_queue and card_index in range(len(self.discover_queue[0])) and not self.dead

    def summon_from_void(self, monster: MonsterCard):
        if self.room_on_board():
            self.gain_board_card(monster)
            self.broadcast_buy_phase_event(events.SummonBuyEvent(monster))

    def room_on_board(self):
        return len(self.in_play) < self.maximum_board_size

    def draw(self, unfreeze: Optional[bool] = True):
        self.return_cards(unfreeze)
        number_of_cards = (3 + self.tavern_tier // 2 - len(self.store))
        number_of_cards = min(number_of_cards, self.maximum_store_size - self.store_size())
        self.extend_store(self.tavern.deck.draw(self, number_of_cards))

    def purchase(self, index: StoreIndex):
        # check if the index is valid
        assert self.valid_purchase(index)
        card = self.pop_store_card(index)
        self.coins -= self.minion_cost
        card.frozen = False
        self._hand.append(card)
        event = events.BuyEvent(card)
        self.broadcast_buy_phase_event(event)
        self.check_golden(type(card))
        self.purchased_minions.append(type(card))

    def valid_purchase(self, index: 'StoreIndex') -> bool:
        if self.dead:
            return False
        if self.discover_queue:
            return False
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
                    self._in_play.remove(card)
                if card in self.hand:
                    self._hand.remove(card)
            golden_card = check_card()
            golden_card.golden_transformation(cards)
            self.gain_hand_card(golden_card)

    def reroll_store(self):
        assert self.valid_reroll()
        if self.free_refreshes >= 1:
            self.free_refreshes -= 1
        else:
            self.coins -= self.refresh_store_cost
        self.draw()
        self.broadcast_buy_phase_event(events.RefreshStoreEvent())

    def valid_reroll(self) -> bool:
        if self.dead:
            return False
        if self.discover_queue:
            return False
        return self.coins >= self.refresh_store_cost or self.free_refreshes >= 1

    def return_cards(self, unfreeze: Optional[bool] = True):
        if unfreeze:
            self.unfreeze()
        self.tavern.deck.return_cards(
            itertools.chain.from_iterable([card.dissolve() for card in self.store if not card.frozen]))
        self._store = [card for card in self.store if card.frozen]
        self.unfreeze()

    def freeze(self):
        for card in self.store:
            card.frozen = True

    def unfreeze(self):
        for card in self.store:
            card.frozen = False

    def sell_minion(self, index: BoardIndex):
        assert self.valid_sell_minion(index)
        card = self.in_play[index]
        self.broadcast_buy_phase_event(events.SellEvent(card))
        self._in_play.remove(card)
        self.coins += card.redeem_rate
        returned_cards = card.dissolve()
        self.tavern.deck.return_cards(returned_cards)

    def valid_sell_minion(self, index: 'BoardIndex') -> bool:
        if self.dead:
            return False
        if self.discover_queue:
            return False
        return self.valid_board_index(index)

    def hero_power(self, board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        self.hero.hero_power(BuyPhaseContext(self, self.tavern.randomizer), board_index, store_index)

    def valid_hero_power(self, board_target: Optional['BoardIndex'] = None,
                         store_target: Optional['StoreIndex'] = None) -> bool:
        if self.dead:
            return False
        if self.discover_queue:
            return False
        return self.hero.hero_power_valid(BuyPhaseContext(self, self.tavern.randomizer), board_target, store_target)

    def broadcast_buy_phase_event(self, event: 'CardEvent', randomizer: Optional['Randomizer'] = None):
        self.hero.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.in_play.copy():
            if card in self.in_play:
                card.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.hand.copy():
            if card in self.hand:
                card.handle_event_in_hand(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))

    def valid_rearrange_cards(self, permutation: List[int]) -> bool:
        if self.dead:
            return False
        if self.discover_queue:
            return False
        return len(permutation) == len(self.in_play) and set(permutation) == set(range(len(self.in_play)))

    def rearrange_cards(self, permutation: List[int]):
        assert self.valid_rearrange_cards(permutation)
        self._in_play = [self._in_play[i] for i in permutation]

    def hand_size(self):
        return len(self.hand) + len(
            self.triple_rewards) + self.gold_coins + self.bananas + self.hero.occupied_hand_slots()

    def room_in_hand(self):
        return self.hand_size() < self.maximum_hand_size

    def store_size(self):
        return len(self.store) + self.hero.occupied_store_slots()

    def max_tier(self):
        return len(self._tavern_upgrade_costs)

    def choose_hero(self, hero_index: HeroChoiceIndex):
        assert (self.valid_choose_hero(hero_index))
        self.hero = self.hero_options.pop(hero_index)
        self.tavern.hero_pool.extend(self.hero_options)
        self.hero_options = []
        self.health = self.hero.starting_health()
        self.minion_cost = self.hero.minion_cost()
        self.refresh_store_cost = self.hero.refresh_cost()
        self._tavern_upgrade_costs = self.hero.tavern_upgrade_costs()
        self.tavern_upgrade_cost = self.hero.tavern_upgrade_costs()[1]

    def valid_choose_hero(self, hero_index: HeroChoiceIndex):
        return hero_index in range(len(self.hero_options))

    def take_damage(self, damage: int):
        if not any(card.give_immunity for card in self.in_play) and not self.hero.give_immunity:
            self.health -= damage
            self.broadcast_buy_phase_event(events.PlayerDamagedEvent())
            if self.health <= 0:
                self.resolve_death()

    def redeem_gold_coin(self):
        if self.gold_coins >= 1:
            self.gold_coins -= 1
            self.coins += 1

    def gain_hand_card(self, card: 'MonsterCard'):
        if self.room_in_hand():
            self._hand.append(card)
            self.check_golden(type(card))

    def gain_board_card(self, card: 'MonsterCard'):
        if self.room_on_board():
            self._in_play.append(card)
            self.check_golden(type(card))

    def add_to_store(self, card: 'MonsterCard'):
        if self.store_size() < self.maximum_store_size:
            self._store.append(card)
            card.apply_nomi_buff(self)
            self.broadcast_buy_phase_event(events.AddToStoreEvent(card))

    def extend_store(self, cards: List['MonsterCard']):
        for card in cards:
            self.add_to_store(card)

    def remove_hand_card(self, card: 'MonsterCard'):
        self._hand.remove(card)

    def remove_board_card(self, card: 'MonsterCard'):
        self._in_play.remove(card)

    def remove_store_card(self, card: 'MonsterCard'):
        card.frozen = False
        self._store.remove(card)

    def pop_hand_card(self, index: int) -> 'MonsterCard':
        return self._hand.pop(index)

    def pop_board_card(self, index: int) -> 'MonsterCard':
        return self._in_play.pop(index)

    def pop_store_card(self, index: int) -> 'MonsterCard':
        return self._store.pop(index)

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
        assert self.big_bananas <= self.bananas
        self.bananas -= 1
        bonus = 1
        if self.big_bananas > 0:  # for now, big bananas will always be used first
            self.big_bananas -= 1
            bonus = 2
        if board_index is not None:
            self.in_play[board_index].attack += bonus
            self.in_play[board_index].health += bonus
        if store_index is not None:
            self.store[store_index].attack += bonus
            self.store[store_index].health += bonus

    def valid_use_banana(self, board_index: Optional['BoardIndex'] = None,
                         store_index: Optional['StoreIndex'] = None) -> bool:
        if self.dead:
            return False
        if self.discover_queue:
            return False
        if self.bananas <= 0:
            return False
        if board_index == store_index:
            return False
        if board_index is not None and not self.valid_board_index(board_index):
            return False
        if store_index is not None and not self.valid_store_index(store_index):
            return False
        return True

    def resolve_death(self):
        assert not self.dead and self.health <= 0
        self.dead = True
        self.broadcast_global_event(events.PlayerDeadEvent(self))
        self.tavern.deck.return_cards(itertools.chain.from_iterable([card.dissolve() for card in self.in_play]))

    def broadcast_global_event(self, event: 'CardEvent'):
        for player in self.tavern.players.values():
            player.hero.handle_event(event, BuyPhaseContext(player, self.tavern.randomizer))

    def next_opponent(self) -> 'Player':
        for p, o in self.tavern.current_player_pairings:
            if p == self:
                return o
            if o == self:
                return p

    def hero_select_discover(self, discover_index: 'DiscoverIndex'):
        self.hero.select_discover(discover_index)

    def valid_hero_select_discover(self, discover_index: 'DiscoverIndex'):
        if self.dead:
            return False
        return self.hero.valid_select_discover(discover_index)

    def current_build(self) -> Tuple[Optional['MONSTER_TYPES'], Optional[int]]:
        cards_by_type = {monster_type.name: 0 for monster_type in MONSTER_TYPES.single_types()}
        for card in self.in_play:
            if card.monster_type == MONSTER_TYPES.ALL:
                for t in cards_by_type:
                    cards_by_type[t] += 1
            elif card.monster_type is not None:
                cards_by_type[card.monster_type.name] += 1
        ranked = sorted(cards_by_type.items(), key=lambda item: item[1], reverse=True)
        if ranked[0][1] == ranked[1][1]:
            return None, None
        else:
            return ranked[0][0], ranked[0][1]
