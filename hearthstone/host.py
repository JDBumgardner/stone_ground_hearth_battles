import typing
from typing import Dict

from hearthstone.agent import EndPhaseAction
from hearthstone.tavern import Tavern

if typing.TYPE_CHECKING:
    from hearthstone.agent import Agent
import asyncio


class Host:
    def start_game(self):
        pass

    def play_round(self):
        pass

    def game_over(self):
        pass

    def play_game(self):
        pass


class RoundRobinHost(Host):
    tavern: Tavern
    agents: Dict[str, 'Agent']

    def __init__(self, agents: Dict[str, 'Agent']):
        self.tavern = Tavern()
        self.agents = agents
        for player_name in agents.keys():
            self.tavern.add_player(player_name)
        for player_name, player in self.tavern.players.items():
            player.choose_hero(
                asyncio.get_event_loop().run_until_complete(self.agents[player_name].hero_choice_action(player)))

    def play_round_generator(self) -> typing.Generator:  # TODO: think about how to test this code
        self.tavern.buying_step()
        for player_name, player in self.tavern.players.items():
            if player.dead:
                continue
            agent = self.agents[player_name]
            for _ in range(20):
                action = asyncio.get_event_loop().run_until_complete(agent.buy_phase_action(player))
                yield
                action.apply(player)
                if player.discover_queue:
                    discovered_card = asyncio.get_event_loop().run_until_complete(agent.discover_choice_action(player))
                    player.select_discover(discovered_card)

                if type(action) is EndPhaseAction:
                    break
            if len(player.in_play) > 1:
                arrangement = asyncio.get_event_loop().run_until_complete(agent.rearrange_cards(player))
                assert set(arrangement) == set(player.in_play)
                player.in_play = arrangement
        self.tavern.combat_step()
        if self.tavern.game_over():
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                asyncio.get_event_loop().run_until_complete(self.agents[name].game_over(player, position))

    def play_round(self):
        for _ in self.play_round_generator():
            pass

    def game_over(self):
        return self.tavern.game_over()

    def play_game(self):
        self.start_game()
        while not self.game_over():
            self.play_round()


class AsyncHost(Host):
    tavern: Tavern
    agents: Dict[str, 'Agent']

    def __init__(self, agents: Dict[str, 'Agent']):
        self.tavern = Tavern()
        self.agents = agents
        for player_name in agents.keys():
            self.tavern.add_player(player_name)

    def start_game(self):
        asyncio.get_event_loop().run_until_complete(self._async_start_game())

    async def _async_start_game(self):
        player_choices = {}

        async def set_player_choice(player_name, player):
            player_choices[player_name] = await self.agents[player_name].hero_choice_action(player)

        player_choice_tasks = []
        for player_name, player in self.tavern.players.items():
            player_choice_tasks.append(asyncio.create_task(set_player_choice(player_name, player)))
        await asyncio.gather(*player_choice_tasks)

        for player_name, player in self.tavern.players.items():
            player.choose_hero(player_choices[player_name])

    def play_round(self):
        return asyncio.get_event_loop().run_until_complete(self._async_play_round())

    async def _async_play_round(self):
        self.tavern.buying_step()

        async def perform_player_actions(agent, player):
            for _ in range(20):
                if player.discover_queue:
                    discovered_card = await agent.discover_choice_action(player)
                    player.select_discover(discovered_card)
                else:
                    action = await agent.buy_phase_action(player)
                    action.apply(player)
                    if type(action) is EndPhaseAction:
                        break
            if len(player.in_play) > 1:
                arrangement = await agent.rearrange_cards(player)
                player.rearrange_cards(arrangement)

        perform_player_action_tasks = []
        for player_name, player in self.tavern.players.items():
            if player.dead:
                continue
            perform_player_action_tasks.append(
                asyncio.create_task(perform_player_actions(self.agents[player_name], player)))
        await asyncio.gather(*perform_player_action_tasks)

        self.tavern.combat_step()
        if self.tavern.game_over():
            game_over_tasks = []
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                game_over_tasks.append(asyncio.create_task(self.agents[name].game_over(player, position)))
            await asyncio.gather(*game_over_tasks)

    def game_over(self):
        return self.tavern.game_over()

    def play_game(self):
        self.start_game()
        while not self.game_over():
            self.play_round()
