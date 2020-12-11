import typing
from typing import List
from hearthstone.simulator.agent import generate_valid_actions, BuyAction, EndPhaseAction, SummonAction, SellAction, \
    DiscoverChoiceAction, RearrangeCardsAction, HeroDiscoverAction
from hearthstone.simulator.agent import  TavernUpgradeAction, RerollAction
from hearthstone.battlebots.bot_types import PriorityFunctionBot

from hearthstone.simulator.core.player import Player, StoreIndex

if typing.TYPE_CHECKING:
    from hearthstone.simulator.agent import StandardAction


class PriorityBot(PriorityFunctionBot):
    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    async def buy_phase_action(self, player: 'Player') -> 'StandardAction':
        all_actions = list(generate_valid_actions(player))

        upgrade_action = TavernUpgradeAction()
        if upgrade_action.valid(player):
            return upgrade_action

        top_hand_priority = max([self.priority(player, card) for card in player.hand], default=None)
        top_store_priority = max([self.priority(player, card) for card in player.store], default=None)
        bottom_board_priority = min([self.priority(player, card) for card in player.in_play], default=None)

        if top_hand_priority:
            if player.room_on_board():
                return [action for action in all_actions if type(action) is SummonAction and self.priority(player, player.hand[action.index]) == top_hand_priority][0]
            else:
                if top_hand_priority > bottom_board_priority:
                    return [action for action in all_actions if type(action) is SellAction and self.priority(player, player.in_play[action.index]) == bottom_board_priority][0]

        if top_store_priority:
            if player.room_on_board() or bottom_board_priority < top_store_priority:
                buy_action = BuyAction([StoreIndex(i) for i, card in enumerate(player.store) if self.priority(player, card) == top_store_priority][0])
                if buy_action.valid(player):
                    return buy_action

        reroll_action = RerollAction()
        if reroll_action.valid(player):
            return reroll_action

        return EndPhaseAction(False)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        discover_cards = player.discover_queue[0]
        discover_cards = sorted(discover_cards, key=lambda card: self.priority(player, card), reverse=True)
        return DiscoverChoiceAction(player.discover_queue[0].index(discover_cards[0]))

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        return HeroDiscoverAction(self.local_random.choice(range(len(player.hero.discover_choices))))
