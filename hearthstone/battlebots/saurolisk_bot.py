import random
import typing

from hearthstone.simulator.agent.actions import StandardAction, generate_standard_actions, BuyAction, EndPhaseAction, \
    SummonAction, \
    SellAction, TavernUpgradeAction, RerollAction, DiscoverChoiceAction, RearrangeCardsAction, \
    FreezeDecision
from hearthstone.simulator.agent.agent import Agent
from hearthstone.simulator.core.card_pool import RabidSaurolisk
from hearthstone.simulator.core.player import Player, BoardIndex

if typing.TYPE_CHECKING:
    pass


class SauroliskBot(Agent):
    authors = ["Jake Bumgardner"]

    def __init__(self, seed: int):
        self.local_random = random.Random(seed)

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        permutation = list(range(len(player.in_play)))
        self.local_random.shuffle(permutation)
        return RearrangeCardsAction(permutation)

    @staticmethod
    def desired_card(card):
        return type(card) == RabidSaurolisk or card.deathrattles

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        all_actions = list(generate_standard_actions(player))

        upgrade_actions = [action for action in all_actions if type(action) is TavernUpgradeAction]
        if upgrade_actions:
            return upgrade_actions[0]

        summon_actions = [action for action in all_actions if type(action) is SummonAction]
        if summon_actions:
            return summon_actions[0]

        buy_actions = [action for action in all_actions if
                       type(action) is BuyAction and self.desired_card(player.store[action.index])]
        if buy_actions:
            return buy_actions[0]

        reroll_action = RerollAction()
        if reroll_action.valid(player):
            return reroll_action

        if len(player.in_play) == 7:
            for index, card in enumerate(player.in_play):
                if type(card) is not RabidSaurolisk:
                    return SellAction(BoardIndex(index))

        return EndPhaseAction(FreezeDecision.NO_FREEZE)

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        discover_cards = player.discover_queue[0].items
        discover_cards = sorted(discover_cards, key=lambda card: self.desired_card(card), reverse=True)
        return DiscoverChoiceAction(player.discover_queue[0].items.index(discover_cards[0]))
