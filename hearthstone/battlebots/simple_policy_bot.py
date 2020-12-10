import json
import math
import random
import typing
from collections import defaultdict
from typing import List, Optional

from hearthstone.simulator.agent import Agent, generate_valid_actions, TavernUpgradeAction, RerollAction, \
    EndPhaseAction, \
    SellAction, StandardAction, BuyAction, SummonAction, DiscoverChoiceAction, RearrangeCardsAction, HeroDiscoverAction

if typing.TYPE_CHECKING:

    from hearthstone.simulator.core.player import Player


class SimplePolicyBot(Agent):

    def __init__(self, authors: List[str], seed: int):
        if not authors:
            authors = ["Jeremy Salwen", "Jacob Bumgardner"]
        self.authors = authors
        self.priority_buy_dict = defaultdict(lambda: 0.0)
        self.priority_summon_dict = defaultdict(lambda: 0.0)
        self.priority_sell_dict = defaultdict(lambda: 0.0)
        self.reroll_priority = 0.0
        self.local_random = random.Random(seed)
        self.current_game_buy = defaultdict(lambda: 0.0)
        self.current_game_summon = defaultdict(lambda: 0.0)
        self.current_game_sell = defaultdict(lambda: 0.0)
        self.current_game_reroll = 0.0
        self.number_of_games = 0
        self.average_placement = 0.0

    def learn_from_game(self, place: int, learning_rate: float):
        for card, score in self.current_game_buy.items():
            self.priority_buy_dict[card] += (self.average_placement-place) * score * learning_rate
        for card, score in self.current_game_summon.items():
            self.priority_summon_dict[card] += (self.average_placement-place) * score * learning_rate
        for card, score in self.current_game_sell.items():
            self.priority_sell_dict[card] += (self.average_placement - place) * score * learning_rate
        self.reroll_priority += (self.average_placement - place) * score * learning_rate
        self.current_game_buy = defaultdict(lambda: 0.0)
        self.current_game_summon = defaultdict(lambda: 0.0)
        self.current_game_sell = defaultdict(lambda: 0.0)
        self.average_placement = (self.average_placement * self.number_of_games + place) / (self.number_of_games +1)
        self.number_of_games += 1

    def save_to_file(self, path):
        with open(path, "w") as f:
            json.dump(self.priority_dict, f)

    def read_from_file(self, path):
        with open(path) as f:
            self.priority_dict.update(json.load(f))

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        all_actions = list(generate_valid_actions(player))

        if player.tavern_tier < 2:
            upgrade_action = TavernUpgradeAction()
            if upgrade_action.valid(player):
                return upgrade_action

        ranked_actions = [(self.score_action(player, action), action) for action in all_actions]
        ranked_actions = [(score, action) for score, action in ranked_actions if score is not None]
        choice = self.local_random.choices(ranked_actions, weights=[math.exp(score) for score, _ in ranked_actions])
        self.update_gradient(player, choice[0], ranked_actions)
        return choice[0][1]

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        discover_cards = player.discover_queue[0]
        discover_cards = sorted(discover_cards, key=lambda card: self.priority_buy_dict[type(card).__name__], reverse=True)
        return DiscoverChoiceAction(player.discover_queue[0].index(discover_cards[0]))

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        return HeroDiscoverAction(self.local_random.choice(range(len(player.hero.discover_choices))))

    def score_action(self, player: Player, action: StandardAction) -> Optional[float]:
        if type(action) is BuyAction:
            return self.priority_buy_dict[type(player.store[action.index]).__name__]
        if type(action) is SellAction:
            return self.priority_sell_dict[type(player.in_play[action.index]).__name__]
        if type(action) is SummonAction:
            return self.priority_summon_dict[type(action.card).__name__]
        if type(action) is EndPhaseAction:
            return 0
        if type(action) is RerollAction:
            return self.reroll_priority
        return None

    def update_gradient(self, player, choice, ranked_actions):
        exponentials = [math.exp(score) for score, _ in ranked_actions]
        softmax = [score / sum(exponentials) for score in exponentials]
        chosen_index = ranked_actions.index(choice)
        gradients = [-1 * x * softmax[chosen_index] for x in softmax]
        gradients[chosen_index] = softmax[chosen_index]*(1 - softmax[chosen_index])
        for (score, action), gradient in zip(ranked_actions, gradients):
            if type(action) is BuyAction:
                self.current_game_buy[type(player.store[action.index]).__name__] += gradient
            if type(action) is SummonAction:
                self.current_game_summon[type(action.card).__name__] += gradient
            if type(action) is SellAction:
                self.current_game_sell[type(player.in_play[action.index]).__name__] += gradient
            if type(action) is RerollAction:
                self.current_game_reroll += gradient
