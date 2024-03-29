import itertools
import typing
from typing import Optional, List, Callable, Type, Tuple

from frozenlist.frozen_list import FrozenList
from hearthstone.simulator.core import events
from hearthstone.simulator.core.cards import MonsterCard, CardLocation
from hearthstone.simulator.core.discover_object import DiscoverObject, Discoverable, DiscoverType
from hearthstone.simulator.core.events import BuyPhaseContext, CardEvent
from hearthstone.simulator.core.hero import EmptyHero
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.secrets import Secret
from hearthstone.simulator.core.spell import Spell
from hearthstone.simulator.core.spell_pool import TripleRewardCard, TheUnlimitedCoin, BloodGem

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.tavern import Tavern
    from hearthstone.simulator.core.hero import Hero
    from hearthstone.simulator.core.randomizer import Randomizer


TEST_MODE = False

StoreIndex = typing.NewType("StoreIndex", int)
HandIndex = typing.NewType("HandIndex", int)
SpellIndex = typing.NewType("SpellIndex", int)
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
        self.discover_queue: List['DiscoverObject'] = []
        self.maximum_board_size = 7
        self.maximum_hand_size = 10
        self.maximum_store_size = 7
        self.refresh_store_cost = 1
        self._tavern_upgrade_costs = (0, 5, 7, 8, 9, 10)
        self._tavern_upgrade_cost = 5
        self.tavern_cost_reduction = 1
        self._hand: List[MonsterCard] = []
        self._in_play: List[MonsterCard] = []
        self._store: List[MonsterCard] = []
        self._spells: List[Spell] = []
        self.minion_cost = 3
        self.purchased_minions: List['Type'] = []
        self.played_minions: List['Type'] = []
        self.last_opponent_warband: List['MonsterCard'] = []
        self.dead = False
        self.nomi_bonus = 0
        self._free_refreshes = 0
        self.new_recruit = False
        self.the_good_stuff = False
        self.the_unlimited_coins_played = 0
        self.battlecry_multiplier = 1
        self.num_turn_start_free_refreshes = 0
        self.secrets: List['Secret'] = []

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
        if TEST_MODE:
            return FrozenList(self._in_play)
        else:
            return self._in_play

    @property
    def hand(self) -> List[MonsterCard]:
        if TEST_MODE:
            return FrozenList(self._hand)
        else:
            return self._hand

    @property
    def store(self) -> List[MonsterCard]:
        if TEST_MODE:
            return FrozenList(self._store)
        else:
            return self._store

    @property
    def spells(self) -> List[Spell]:
        if TEST_MODE:
            return FrozenList(self._spells)
        else:
            return self._spells

    @staticmethod
    def new_player_with_hero(tavern: Optional['Tavern'], name: str, hero: Optional['Hero'] = None) -> 'Player':
        if hero is None:
            hero = EmptyHero()
        player = Player(tavern, name, [hero])
        player.choose_hero_from_index(HeroChoiceIndex(0))
        return player

    @property
    def coin_income_rate(self):
        return min(self.tavern.turn_count + 3, 10)

    @property
    def free_refreshes(self):
        return self._free_refreshes

    def set_free_refreshes(self, num: int):
        self._free_refreshes = max(num, self._free_refreshes)

    def buying_step(self):
        self.reset_purchased_minions_list()
        self.reset_played_minions_list()
        self.apply_turn_start_income()
        self.apply_darkmoon_prize_effects()
        self.draw(unfreeze=False)
        self.hero.on_buy_step()
        self.broadcast_buy_phase_event(events.BuyStartEvent())

    def at_buy_end(self):
        self.decrease_tavern_upgrade_cost()
        self.return_unlimited_coins()

    def reset_purchased_minions_list(self):
        self.purchased_minions = []

    def reset_played_minions_list(self):
        self.played_minions = []

    def apply_turn_start_income(self):
        self.coins = self.coin_income_rate

    def apply_darkmoon_prize_effects(self):
        self.battlecry_multiplier = 1
        self.set_free_refreshes(self.num_turn_start_free_refreshes)

    def decrease_tavern_upgrade_cost(self):
        self.tavern_upgrade_cost -= self.tavern_cost_reduction

    def return_unlimited_coins(self):
        for _ in range(self.the_unlimited_coins_played):
            self.gain_spell(TheUnlimitedCoin())
        self.the_unlimited_coins_played = 0

    def upgrade_tavern(self):
        assert self.valid_upgrade_tavern()
        self.coins -= self.tavern_upgrade_cost
        self.tavern_tier += 1
        if self.tavern_tier < self.max_tier():
            self.tavern_upgrade_cost = self._tavern_upgrade_costs[self.tavern_tier]
        self.broadcast_buy_phase_event(events.TavernUpgradeEvent())

    def valid_upgrade_tavern(self) -> bool:
        if not self.valid_standard_action():
            return False
        return self.base_valid_upgrade_tavern()

    def base_valid_upgrade_tavern(self) -> bool:
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
            self.gain_spell(TripleRewardCard(min(self.tavern_tier + 1, 6)))
        target_cards = [self.in_play[target] for target in targets]
        self.broadcast_buy_phase_event(events.SummonBuyEvent(card, target_cards))
        self.played_minions.append(type(card))

    def valid_summon_from_hand(self, index: HandIndex, targets: Optional[List[BoardIndex]] = None) -> bool:
        if not self.valid_standard_action():
            return False
        return self.base_valid_summon_from_hand(index, targets)

    def base_valid_summon_from_hand(self, index: HandIndex, targets: Optional[List[BoardIndex]] = None) -> bool:
        if not self.base_room_to_summon(index):
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
        if not self.valid_standard_action():
            return False
        return self.base_room_to_summon(index)

    def base_room_to_summon(self, index: HandIndex):
        if not self.valid_hand_index(index):
            return False
        if not self.room_on_board():
            return False
        return True

    # TODO: Jarett help make discoverables unique are cards with more copies in the deck more likely to be discovered?
    def draw_discover(self, predicate: Callable[['MonsterCard'], bool], num: int = 3, discover_function: Optional[Callable[['Discoverable'], None]] = None, dissolve: bool = True):
        # Jeremy says: Hmm, we can run out of unique cards.  Changed to be all cards for now.
        discoverables = [card for card in self.tavern.deck.all_cards() if predicate(card)]
        discovered_cards = []
        for _ in range(num):
            discovered_cards.append(self.tavern.randomizer.select_discover_card(discoverables))
            discoverables.remove(discovered_cards[-1])
            self.tavern.deck.remove_card(discovered_cards[-1])

        if discover_function is None:
            discover_function = self.gain_hand_card
        self.discover_queue.append(DiscoverObject(discovered_cards, discover_function, dissolve, DiscoverType.CARD))

    def select_discover(self, card_index: 'DiscoverIndex'):
        assert self.valid_select_discover(card_index)
        next_discover = self.discover_queue.pop(0)
        next_discover.select_item(card_index, self)

    def valid_select_discover(self, card_index: 'DiscoverIndex'):
        return self.discover_queue and card_index in range(len(self.discover_queue[0].items)) and not self.dead

    def summon_from_void(self, monster: MonsterCard):
        if self.room_on_board():
            self.gain_board_card(monster, False)
            self.broadcast_buy_phase_event(events.SummonBuyEvent(monster))

    def room_on_board(self):
        return len(self.in_play) < self.maximum_board_size

    def draw(self, unfreeze: Optional[bool] = True, predicate: Optional[Callable] = lambda card: True):
        self.return_cards(unfreeze)
        number_of_cards = (3 + self.tavern_tier // 2 - len(self.store)) + int(self.new_recruit)
        number_of_cards = min(number_of_cards, self.maximum_store_size - self.store_size())
        self.extend_store(self.tavern.deck.draw(self, number_of_cards, predicate))

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
        if not self.valid_standard_action():
            return False
        return self.base_valid_purchase(index)

    def base_valid_purchase(self, index: 'StoreIndex') -> bool:
        if not self.valid_store_index(index):
            return False
        if self.coins < self.minion_cost:
            return False
        if not self.room_in_hand():
            return False
        return True

    def check_golden(self, check_card: Type[MonsterCard]):
        cards = [card for card in self.in_play + self.hand if isinstance(card, check_card) and not card.golden]
        if len(cards) >= 3:
            for card in cards[:3]:
                if card in self.in_play:
                    self._in_play.remove(card)
                if card in self.hand:
                    self._hand.remove(card)
            golden_card = check_card()
            golden_card.golden_transformation(cards)
            self.gain_hand_card(golden_card)

    def reroll_store(self):
        assert self.valid_reroll()
        if self._free_refreshes >= 1:
            self._free_refreshes -= 1
        else:
            self.coins -= self.refresh_store_cost
        self.draw()
        self.broadcast_buy_phase_event(events.RefreshStoreEvent())

    def valid_reroll(self) -> bool:
        if not self.valid_standard_action():
            return False
        return self.base_valid_reroll()

    def base_valid_reroll(self) -> bool:
        return self.coins >= self.refresh_store_cost or self._free_refreshes >= 1

    def return_cards(self, unfreeze: Optional[bool] = True):
        if unfreeze:
            self.unfreeze()
        self.tavern.deck.return_cards(
            itertools.chain.from_iterable((card.dissolve() for card in self.store if not card.frozen)))
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
        if not self.valid_standard_action():
            return False
        return self.base_valid_sell_minion(index)

    def base_valid_sell_minion(self, index: 'BoardIndex') -> bool:
        return self.valid_board_index(index)

    def hero_power(self, board_index: Optional['BoardIndex'] = None, store_index: Optional['StoreIndex'] = None):
        self.hero.hero_power(BuyPhaseContext(self, self.tavern.randomizer), board_index, store_index)

    def valid_hero_power(self, board_target: Optional['BoardIndex'] = None,
                         store_target: Optional['StoreIndex'] = None) -> bool:
        if not self.valid_standard_action():
            return False
        return self.base_valid_hero_power(board_target, store_target)

    def base_valid_hero_power(self, board_target: Optional['BoardIndex'] = None,
                              store_target: Optional['StoreIndex'] = None) -> bool:
        return self.hero.hero_power_valid(BuyPhaseContext(self, self.tavern.randomizer), board_target, store_target)

    def broadcast_buy_phase_event(self, event: 'CardEvent', randomizer: Optional['Randomizer'] = None):
        self.hero.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.in_play.copy():
            if card in self.in_play:
                card.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.hand.copy():
            if card in self.hand:
                card.handle_event_in_hand(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for secret in self.secrets.copy():
            if secret in self.secrets:
                secret.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))

    def valid_rearrange_cards(self, permutation: List[int]) -> bool:
        if not self.valid_standard_action():
            return False
        return len(permutation) == len(self.in_play) and set(permutation) == set(range(len(self.in_play)))

    def rearrange_cards(self, permutation: List[int]):
        assert self.valid_rearrange_cards(permutation), "in play {} permutation {}".format(self.in_play, permutation)
        self._in_play = [self._in_play[i] for i in permutation]

    def hand_size(self):
        return len(self.hand) + len(self.spells) + self.hero.occupied_hand_slots()

    def room_in_hand(self):
        return self.hand_size() < self.maximum_hand_size

    def store_size(self):
        return len(self.store) + self.hero.occupied_store_slots()

    def refresh_size(self):
        return 3 + self.tavern_tier // 2 - self.store_size()

    def max_tier(self):
        return len(self._tavern_upgrade_costs)

    def choose_hero_from_index(self, hero_index: HeroChoiceIndex):
        assert (self.valid_choose_hero(hero_index))
        self.choose_hero(self.hero_options.pop(hero_index))

    def choose_hero(self, hero: 'Hero'):
        self.hero = hero
        self.tavern.hero_pool.extend(self.hero_options)
        self.hero_options = []
        self.health = self.hero.starting_health()
        self.minion_cost = self.hero.minion_cost()
        self.refresh_store_cost = self.hero.refresh_cost()
        self._tavern_upgrade_costs = self.hero.tavern_upgrade_costs()
        if self.tavern.turn_count == 0:
            self.tavern_upgrade_cost = self.hero.tavern_upgrade_costs()[1]

    def valid_choose_hero(self, hero_index: HeroChoiceIndex):
        return hero_index in range(len(self.hero_options))

    def take_damage(self, damage: int):
        if not any(card.give_immunity for card in self.in_play) and not self.hero.give_immunity:
            self.health -= damage
            self.broadcast_buy_phase_event(events.PlayerDamagedEvent())
            if self.health <= 0:
                self.resolve_death()

    def gain_hand_card(self, card: 'MonsterCard'):
        if self.room_in_hand():
            self._hand.append(card)
            self.check_golden(type(card))

    def gain_board_card(self, card: 'MonsterCard', check_for_triples: bool = True):
        if self.room_on_board():
            self._in_play.append(card)
            if check_for_triples:
                self.check_golden(type(card))

    def gain_spell(self, spell: 'Spell'):
        if self.room_in_hand():
            self._spells.append(spell)
            spell.on_gain(BuyPhaseContext(self, self.tavern.randomizer))

    def add_to_store(self, card: 'MonsterCard'):
        if self.store_size() < self.maximum_store_size:
            self._store.append(card)
            card.apply_nomi_buff(self)
            card.health += 2 if self.the_good_stuff else 0
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

    def remove_spell(self, spell: 'Spell'):
        self._spells.remove(spell)

    def pop_hand_card(self, index: int) -> 'MonsterCard':
        return self._hand.pop(index)

    def pop_board_card(self, index: int) -> 'MonsterCard':
        return self._in_play.pop(index)

    def pop_store_card(self, index: int) -> 'MonsterCard':
        return self._store.pop(index)

    def pop_spell(self, index: int) -> 'Spell':
        return self._spells.pop(index)

    def valid_board_index(self, index: 'BoardIndex') -> bool:
        return 0 <= index < len(self.in_play)

    def valid_hand_index(self, index: 'HandIndex') -> bool:
        return 0 <= index < len(self.hand)

    def valid_store_index(self, index: 'StoreIndex') -> bool:
        return 0 <= index < len(self.store)

    def valid_spell_index(self, index: 'SpellIndex') -> bool:
        return 0 <= index < len(self._spells)

    def plus_coins(self, amt: int):
        self.coins += amt

    def resolve_death(self):
        # assert not self.dead and self.health <= 0
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

    def valid_standard_action(self):
        return not self.dead and not self.discover_queue

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

    def valid_play_spell(self, index: 'SpellIndex', board_index: Optional['BoardIndex'] = None,
                         store_index: Optional['StoreIndex'] = None):
        if not self.valid_standard_action():
            return False
        return self.base_valid_play_spell(index, board_index, store_index)

    def base_valid_play_spell(self, index: 'SpellIndex', board_index: Optional['BoardIndex'] = None,
                              store_index: Optional['StoreIndex'] = None):
        if not self.valid_spell_index(index):
            return False
        spell = self.spells[index]
        if self.coins < spell.cost:
            return False
        if not spell.valid(BuyPhaseContext(self, self.tavern.randomizer), board_index, store_index):
            return False
        return True

    def spell_can_be_played(self, index: 'SpellIndex'):
        if not self.valid_spell_index(index):
            return False
        spell = self.spells[index]
        if self.coins < spell.cost:
            return False
        if CardLocation.STORE in spell.target_location:
            for index, card in enumerate(self.store):
                if spell.valid_target(BuyPhaseContext(self, self.tavern.randomizer), store_index=StoreIndex(index)):
                    return True
        if CardLocation.BOARD in spell.target_location:
            for index, card in enumerate(self.in_play):
                if spell.valid_target(BuyPhaseContext(self, self.tavern.randomizer), board_index=BoardIndex(index)):
                    return True
        if spell.target_location is None:
            return True
        return False

    def play_spell(self, index: 'SpellIndex', board_index: Optional['BoardIndex'] = None,
                   store_index: Optional['StoreIndex'] = None):
        assert self.valid_play_spell(index, board_index, store_index)
        spell = self.pop_spell(index)
        self.coins -= spell.cost
        spell.on_play(BuyPhaseContext(self, self.tavern.randomizer), board_index, store_index)

    def play_blood_gem(self, target: 'MonsterCard'):
        board_index = self.in_play.index(target)
        BloodGem().on_play(BuyPhaseContext(self, self.tavern.randomizer), board_index=board_index)

    def swap_hero(self, new_hero: 'Hero'):
        self.hero = new_hero
        self.minion_cost = new_hero.minion_cost()
        self.refresh_store_cost = new_hero.refresh_cost()
        self._tavern_upgrade_costs = new_hero.tavern_upgrade_costs()
