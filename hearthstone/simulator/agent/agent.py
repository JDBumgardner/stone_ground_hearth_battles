from typing import Any

from hearthstone.simulator.agent.actions import HeroChoiceAction, DiscoverChoiceAction, \
    RearrangeCardsAction, StandardAction, HeroDiscoverAction
from hearthstone.simulator.core.player import HeroChoiceIndex
from hearthstone.simulator.core.tavern import Player

Annotation = Any


class AnnotatingAgent:
    async def hero_choice_action(self, player: 'Player') -> HeroChoiceAction:
        return HeroChoiceAction(HeroChoiceIndex(0))

    async def annotated_rearrange_cards(self, player: 'Player') -> (RearrangeCardsAction, Annotation):
        """
        here the player selects a card arrangement one time per combat directly preceding combat

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:  A tuple containing an arrangement of the player's board, and the Agent Annotation to attach to the replay.

        """
        pass

    async def annotated_buy_phase_action(self, player: 'Player') -> (StandardAction, Annotation):
        """
        here the player chooses a buy phase action including:
        purchasing a card from the store
        summoning a card from hand to in_play
        selling a card from hand or from in_play
        and ending the buy phase

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:
            A tuple containing the Action, and the Agent Annotation to attach to the replay.

        """
        pass

    async def annotated_discover_choice_action(self, player: 'Player') -> (DiscoverChoiceAction, Annotation):
        """

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:
            Tuple of MonsterCard to discover, and Annotation to attach to the action.
        """
        pass

    async def annotated_hero_discover_action(self, player: 'Player') -> ('HeroDiscoverAction', Annotation):
        """

        Args:
            player: The player object controlled by this agent. This function should not modify it.

        Returns:
            Tuple of object to discover, and Annotation to attach to the action.
        """
        pass

    async def game_over(self, player: 'Player', ranking: int) -> Annotation:
        """
        Notifies the agent that the game is over and the agent has achieved a given rank
        :param ranking: Integer index 0 to 7 of where the agent placed
        :return:
        """
        pass


class Agent(AnnotatingAgent):

    async def buy_phase_action(self, player: 'Player') -> StandardAction:
        pass

    async def annotated_buy_phase_action(self, player: 'Player') -> (StandardAction, Annotation):
        return await self.buy_phase_action(player), None

    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        pass

    async def annotated_rearrange_cards(self, player: 'Player') -> (RearrangeCardsAction, Annotation):
        return await self.rearrange_cards(player), None

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        pass

    async def annotated_discover_choice_action(self, player: 'Player') -> (DiscoverChoiceAction, Annotation):
        return await self.discover_choice_action(player), None

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        pass

    async def annotated_hero_discover_action(self, player: 'Player') -> ('HeroDiscoverAction', Annotation):
        return await self.hero_discover_action(player), None
