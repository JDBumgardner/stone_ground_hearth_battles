import random
import typing
from typing import List, Callable

from hearthstone.simulator.agent.actions import StandardAction, generate_standard_actions, BuyAction, EndPhaseAction, \
    SummonAction, \
    SellAction, TavernUpgradeAction, RerollAction, DiscoverChoiceAction, RearrangeCardsAction, \
    FreezeDecision
from hearthstone.simulator.agent.agent import Agent
from hearthstone.simulator.core.card_pool import MurlocTidehunter, AlleyCat
from hearthstone.simulator.core.player import Player, StoreIndex, BoardIndex

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.cards import MonsterCard


class EarlyGameBot(Agent):
    def __init__(self, authors: List[str], priority: Callable[['Player', 'MonsterCard'], float], seed: int):
        if not authors:
            authors = ["Jake Bumgardner", "Adam Salwen", "Ethan Saxenian"]
        self.authors = authors
        self.priority = priority
        self.local_random = random.Random(seed)

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        all_actions = list(generate_standard_actions(player))

        if player.tavern.turn_count == 0:
            token_store_index = [StoreIndex(player.store.index(card)) for card in player.store if
                                 type(card) is MurlocTidehunter or type(card) is AlleyCat]
            if token_store_index:
                buy_action = BuyAction(token_store_index[0])
                if buy_action.valid(player):
                    return buy_action

        if player.tavern.turn_count == 2:
            if player.tavern_tier == 2:
                token_board_index = [BoardIndex(player.in_play.index(card)) for card in player.in_play if card.token]
                if token_board_index and player.coins == 5:
                    return SellAction(token_board_index[0])

        if player.tavern.turn_count != 3:
            upgrade_action = TavernUpgradeAction()
            if upgrade_action.valid(player):
                return upgrade_action

        top_hand_priority = max([self.priority(player, card) for card in player.hand], default=None)
        top_store_priority = max([self.priority(player, card) for card in player.store], default=None)
        bottom_board_priority = min([self.priority(player, card) for card in player.in_play], default=None)

        if top_hand_priority:
            if player.room_on_board():
                return [
                    action for action in all_actions
                    if type(action) is SummonAction and self.priority(player,
                                                                      player.hand[action.index]) == top_hand_priority
                ][0]
            else:
                if top_hand_priority > bottom_board_priority:
                    return [
                        action for action in all_actions
                        if type(action) is SellAction and self.priority(player, player.in_play[
                            action.index]) == bottom_board_priority
                    ][0]

        if top_store_priority:
            if player.room_on_board() or bottom_board_priority < top_store_priority:
                buy_action = BuyAction(
                    [StoreIndex(index) for index, card in enumerate(player.store) if
                     self.priority(player, card) == top_store_priority][0]
                )
                if buy_action.valid(player):
                    return buy_action

        reroll_action = RerollAction()
        if reroll_action.valid(player):
            return reroll_action

        return EndPhaseAction(FreezeDecision.NO_FREEZE)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        discover_cards = player.discover_queue[0].items
        discover_cards = sorted(discover_cards, key=lambda card: self.priority(player, card), reverse=True)
        return DiscoverChoiceAction(player.discover_queue[0].items.index(discover_cards[0]))
