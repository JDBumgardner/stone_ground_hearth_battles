import json
import random
import typing
from collections import defaultdict
from typing import List

from hearthstone.simulator.agent import Agent, generate_valid_actions, TavernUpgradeAction, RerollAction, \
    EndPhaseAction, \
    SellAction, StandardAction, BuyAction, SummonAction, DiscoverChoiceAction, RearrangeCardsAction, HeroDiscoverAction, \
    FreezeDecision

if typing.TYPE_CHECKING:

    from hearthstone.simulator.core.player import Player, StoreIndex


class LearnedPriorityBot(Agent):

    def __init__(self, authors: List[str], rand_factor: float, seed: int):
        if not authors:
            authors = ["Jeremy Salwen"]
        self.authors = authors
        self.priority_dict = defaultdict(lambda: 0)
        self.priority = None
        self.set_priority_function()
        self.local_random = random.Random(seed)
        self.rand_factor = rand_factor
        self.current_game_cards = defaultdict(lambda:0)

    def learn_from_game(self, place: int):
        for card, score in self.current_game_cards.items():
            self.priority_dict[card] += (3-place) * score

        self.current_game_cards = defaultdict(lambda:0)

    def set_priority_function(self):
        self.priority = lambda player, card: self.priority_dict[type(card).__name__]

    def save_to_file(self, path):
        with open(path, "w") as f:
            json.dump(self.priority_dict, f)

    def read_from_file(self, path):
        with open(path) as f:
            self.priority_dict.update(json.load(f))
        self.set_priority_function()

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    def adjusted_priority(self, player, card):
        score = self.priority(player, card)
        num_existing = len([existing for existing in player.hand + player.in_play if type(existing) == type(card) and not existing.golden])
        if num_existing == 2:
            score += 100000
        elif num_existing == 1:
            score += 300
        score += 100 * (card.health + card.attack + card.tier)
        return score

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        all_actions = list(generate_valid_actions(player))

        if player.tavern_tier < 2:
            upgrade_action = TavernUpgradeAction()
            if upgrade_action.valid(player):
                return upgrade_action

        top_hand_priority = max([self.adjusted_priority(player, card) for card in player.hand], default=None)
        top_store_priority = max([self.adjusted_priority(player, card) for card in player.store], default=None)
        bottom_board_priority = min([self.adjusted_priority(player, card) for card in player.in_play], default=None)

        if top_hand_priority is not None:
            if player.room_on_board():
                return [action for action in all_actions if type(action) is SummonAction and self.adjusted_priority(player, action.card) == top_hand_priority][0]
            else:
                return [action for action in all_actions if type(action) is SellAction and self.adjusted_priority(player, player.in_play[action.index]) == bottom_board_priority][0]

        if top_store_priority is not None:
            force_buy = False
            if self.local_random.random() < self.rand_factor:
                top_store_priority = self.adjusted_priority(player, self.local_random.choice(player.store))
                force_buy = True
            if player.room_on_board() or bottom_board_priority < top_store_priority or force_buy:
                buy_action = BuyAction([StoreIndex(i) for i, card in enumerate(player.store) if
                                        self.priority(player, card) == top_store_priority][0])
                if buy_action.valid(player):
                    self.current_game_cards[type(buy_action.card).__name__] += 3
                    for card in player.store:
                        self.current_game_cards[type(card).__name__] -= 1
                    return buy_action

        reroll_action = RerollAction()
        if reroll_action.valid(player):
            return reroll_action

        return EndPhaseAction(FreezeDecision.NO_FREEZE)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        discover_cards = player.discover_queue[0]
        discover_cards = sorted(discover_cards, key=lambda card: self.adjusted_priority(card), reverse=True)
        return DiscoverChoiceAction(player.discover_queue[0].index(discover_cards[0]))

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        return HeroDiscoverAction(self.local_random.choice(range(len(player.hero.discover_choices))))
