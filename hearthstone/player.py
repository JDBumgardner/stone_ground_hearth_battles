from typing import Optional, List, Callable
import itertools

from hearthstone.cards import MonsterCard, CardEvent, CardType, Card
from hearthstone.events import SUMMON_BUY, BuyPhaseContext, SELL
from hearthstone.hero import Hero, EmptyHero
from hearthstone.triple_reward_card import TripleRewardCard


class BuyPhaseEvent:
    pass


class Player:
    def __init__(self, tavern: 'Tavern', name: str, hero: Hero = None):
        self.name = name
        self.tavern = tavern
        self.hero = hero or EmptyHero()
        self.health = self.hero.starting_health()
        self.tavern_tier = 1
        self.coins = 0
        self.triple_rewards = []
        self.discovered_cards = []
        self.maximum_board_size = 7
        self.refresh_store_cost = 1
        self.redeem_minion_rate = 1
        self._tavern_upgrade_costs = (0, 5, 7, 8, 9, 10)
        self.hand: List[MonsterCard] = []
        self.in_play: List[MonsterCard] = []
        self.store: List[MonsterCard] = []
        self.frozen = False

    @property
    def coin_income_rate(self):
        return self.tavern.turn_count + 3

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

    @property
    def tavern_upgrade_cost(self):
        return self._tavern_upgrade_costs[self.tavern_tier]

    def apply_turn_start_income(self):
        self.coins = self.coin_income_rate

    def upgrade_tavern(self):
        if self.coins < self.tavern_upgrade_cost:
            print("I can't let you do that Dave")
            return
        self.coins -= self.tavern_upgrade_cost
        self.tavern_tier += 1

    def summon_from_hand(self, monster: MonsterCard, event_target: Optional[MonsterCard] = None,
                         secondary_target: Optional[MonsterCard] = None):
        #  TODO: Validity check for board size
        #  TODO: make sure that the ordering of monster in hand and monster.battlecry are correct
        assert self.room_on_board()
        assert monster in self.hand
        self.hand.remove(monster)
        self.in_play.append(monster)
        if monster.golden:
            self.triple_rewards.append(TripleRewardCard(min(self.tavern_tier + 1, 6)))
        self.broadcast_buy_phase_event(CardEvent(monster, SUMMON_BUY))

    def play_triple_rewards(self):
        if not self.triple_rewards:
            return
        discover_tier = self.triple_rewards.pop(-1).level
        self.draw_discover(lambda card: card.tier == discover_tier)

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

    def purchase(self, index: int):
        # check if the index is valid
        # TODO: send event for the purchased creature for triple thing
        if self.coins < self.store[index].coin_cost:
            return
        card = self.store.pop(index)
        self.coins -= card.coin_cost
        self.hand.append(card)
        self.check_golden(type(card))

    def check_golden(self, check_card: CardType):
        cards = [card for card in self.in_play + self.hand if isinstance(card, check_card) and not card.golden]
        if len(cards) == 3:
            for card in cards:
                if card in self.in_play:
                    self.in_play.remove(card)
                if card in self.hand:
                    self.hand.remove(card)
            golden_card = check_card()
            golden_card.golden_transformation(cards)
            self.hand.append(golden_card)
        elif len(cards) > 3:
            raise ZeroDivisionError("fnord")

    def refresh_store(self):
        if self.coins < self.refresh_store_cost:
            print("I can't let you do that, Dave.")
            return
        self.coins -= self.refresh_store_cost
        self.return_cards()
        self.draw()

    def return_cards(self):
        self.tavern.deck.cards += itertools.chain.from_iterable([card.dissolve() for card in self.store])
        self.store = []

    def freeze(self):
        self.frozen = True

    def sell_minion(self, card: MonsterCard):
        assert card in self.in_play + self.hand
        self.broadcast_buy_phase_event(CardEvent(card, SELL))
        if card in self.hand:
            self.hand.remove(card)
        elif card in self.in_play:
            self.in_play.remove(card)
        self.coins += self.redeem_minion_rate
        self.tavern.deck.cards.append(type(card)())

    def hero_power(self, randomizer: Optional['Randomizer'] = None):
        self.hero.hero_power(BuyPhaseContext(self, randomizer or self.tavern.randomizer))

    def broadcast_buy_phase_event(self, event: CardEvent, randomizer: Optional['Randomizer'] = None):
        for card in self.in_play:
            card.handle_event(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
        for card in self.hand:
            card.handle_event_in_hand(event, BuyPhaseContext(self, randomizer or self.tavern.randomizer))
