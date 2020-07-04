from typing import Optional, List, Callable
import itertools

from hearthstone.cards import MonsterCard, CardEvent, CardType, Card
from hearthstone.events import SUMMON_BUY, BuyPhaseContext, SELL, BUY
from hearthstone.hero import Hero, EmptyHero
from hearthstone.triple_reward_card import TripleRewardCard
from collections import defaultdict


class BuyPhaseEvent:
    pass


class Player:
    def __init__(self, tavern: 'Tavern', name: str, hero_options: List[Hero]):
        self.name = name
        self.tavern = tavern
        self.hero = None
        self.hero_options = hero_options
        self.health = None
        self.tavern_tier = 1
        self.coins = 0
        self.triple_rewards = []
        self.discovered_cards: List[MonsterCard] = []
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

    @staticmethod
    def new_player_with_hero(tavern: 'Tavern', name: str, hero: Hero=None) -> 'Player':
        if hero is None:
            hero = EmptyHero()
        player = Player(tavern, name, [hero])
        player.choose_hero(hero)
        return player

    @property
    def coin_income_rate(self):
        return min(self.tavern.turn_count + 3, 10)

    def player_main_step(self):
        self.draw()
        #  player can:
        #  rearrange monsters
        #  summon monsters
        #  buy from the store
        #  freeze the store
        #  refresh the store
        #  sell monsters
        #  set fight ready

    def apply_turn_start_income(self):
        self.coins = self.coin_income_rate

    def upgrade_tavern(self):
        assert self.validate_upgrade_tavern()
        self.coins -= self.tavern_upgrade_cost
        self.tavern_tier += 1
        if self.tavern_tier < self.max_tier():
            self.tavern_upgrade_cost = self._tavern_upgrade_costs[self.tavern_tier]

    def validate_upgrade_tavern(self) -> bool:
        if self.tavern_tier >= self.max_tier():
            return False
        if self.coins < self.tavern_upgrade_cost:
            return False
        return True

    def summon_from_hand(self, card: MonsterCard, targets: Optional[List[MonsterCard]] = None):
        #  TODO: make sure that the ordering of monster in hand and monster.battlecry are correct
        #  TODO: Jarett can monster be event target
        if targets is None:
            targets = []
        assert self.validate_summon_from_hand(card, targets)
        self.hand.remove(card)
        self.in_play.append(card)
        if card.golden:
            self.triple_rewards.append(TripleRewardCard(min(self.tavern_tier + 1, 6)))
        self.broadcast_buy_phase_event(CardEvent(card, SUMMON_BUY, targets))

    def validate_summon_from_hand(self, card: MonsterCard, targets: Optional[List[MonsterCard]] = None) -> bool:
        #  TODO: Jack num_battlesry_targets should only accept 0,1,2
        if card not in self.hand:
            return False
        if not self.room_on_board():
            return False
        if targets is None:
            targets = []
        valid_targets = [target for target in self.in_play if card.validate_battlecry_target(target)]
        num_possible_targets = min(len(valid_targets), card.num_battlecry_targets)
        if len(targets) != num_possible_targets:
            return False
        if len(set(targets)) != len(targets):
            return False
        for target in targets:
            if target not in valid_targets:
                return False
        return True

    def play_triple_rewards(self):
        if not self.triple_rewards:
            return
        discover_tier = self.triple_rewards.pop(-1).level
        self.draw_discover(lambda card: card.tier == discover_tier)

    def validate_triple_rewards(self) -> bool:
        return bool(self.triple_rewards)

    def draw_discover(self, predicate: Callable[[Card], bool]):
        discoverables = [card for card in self.tavern.deck.cards if predicate(card)]
        for _ in range(3):
            self.discovered_cards.append(self.tavern.randomizer.select_discover_card(discoverables))
            discoverables.remove(self.discovered_cards[-1])
            self.tavern.deck.cards.remove(self.discovered_cards[-1])

    def select_discover(self, card: Card):
        assert (card in self.discovered_cards)
        assert (isinstance(card, MonsterCard))  # TODO: discover other card types
        self.discovered_cards.remove(card)
        self.hand.append(card)
        self.tavern.deck.cards += itertools.chain.from_iterable([card.dissolve() for card in self.discovered_cards])
        self.discovered_cards = []
        self.check_golden(type(card))

    def summon_from_void(self, monster: MonsterCard):
        if self.room_on_board():
            self.in_play.append(monster)
            self.check_golden(type(monster))
        self.broadcast_buy_phase_event(CardEvent(monster, SUMMON_BUY))

    def room_on_board(self):
        return len(self.in_play) < self.maximum_board_size

    def draw(self):
        if self.frozen:
            self.frozen = False
            return
        self.return_cards()
        number_of_cards = 3 + self.tavern_tier // 2
        self.store.extend([self.tavern.deck.draw(self) for _ in range(number_of_cards)])

    def purchase(self, card: MonsterCard):
        # check if the index is valid
        assert self.validate_purchase(card)
        self.store.remove(card)
        self.coins -= card.coin_cost
        self.hand.append(card)
        event = CardEvent(card, BUY)
        self.broadcast_buy_phase_event(event)
        self.check_golden(type(card))

    def validate_purchase(self, card: MonsterCard) -> bool:
        if self.coins < card.coin_cost:
            return False
        if not self.room_in_hand():
            return False
        if card not in self.store:
            return False
        return True

    def check_golden(self, check_card: CardType):
        cards = [card for card in self.in_play + self.hand if isinstance(card, check_card) and not card.golden]
        assert len(cards) <= 3, "fnord"
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
        assert self.validate_reroll()
        self.coins -= self.refresh_store_cost
        self.return_cards()
        self.draw()

    def validate_reroll(self) -> bool:
        return self.coins >= self.refresh_store_cost

    def return_cards(self):
        self.tavern.deck.cards += itertools.chain.from_iterable([card.dissolve() for card in self.store])
        self.store = []

    def freeze(self):
        self.frozen = True

    def sell_minion(self, card: MonsterCard):
        assert self.validate_sell_minion(card)
        self.broadcast_buy_phase_event(CardEvent(card, SELL))
        if card in self.hand:
            self.hand.remove(card)
        elif card in self.in_play:
            self.in_play.remove(card)
        self.coins += card.redeem_rate
        self.tavern.deck.cards.append(type(card)())

    def validate_sell_minion(self, card: MonsterCard) -> bool:
        return card in self.in_play + self.hand

    def hero_power(self):
        self.hero.hero_power(BuyPhaseContext(self, self.tavern.randomizer))

    def validate_hero_power(self) -> bool:
        return self.hero.hero_power_valid(BuyPhaseContext(self, self.tavern.randomizer))

    def broadcast_buy_phase_event(self, event: CardEvent, randomizer: Optional['Randomizer'] = None):
        self.hero.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.in_play.copy():
            card.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.hand.copy():
            card.handle_event_in_hand(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))

    def hand_size(self):
        return len(self.hand) + len(self.triple_rewards)

    def room_in_hand(self):
        return self.hand_size() < self.maximum_hand_size

    def max_tier(self):
        return len(self._tavern_upgrade_costs)

    def choose_hero(self, hero: Hero):
        assert(self.validate_choose_hero(hero))
        self.hero = hero
        self.hero_options = []
        self.health = self.hero.starting_health()

    def validate_choose_hero(self, hero: Hero):
        return self.hero is None and hero in self.hero_options