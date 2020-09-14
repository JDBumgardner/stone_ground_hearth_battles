import collections
import typing
from typing import Dict
from hearthstone.tavern import Tavern
from hearthstone.agent import EndPhaseAction
if typing.TYPE_CHECKING:
    from hearthstone.agent import Agent
import trio


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
            player.choose_hero(trio.run(self.agents[player_name].hero_choice_action, player))

    def play_round_generator(self) -> typing.Generator: # TODO: think about how to test this code
        self.tavern.buying_step()
        for player_name, player in self.tavern.players.items():
            if player.health <= 0:
                continue
            agent = self.agents[player_name]
            for _ in range(20):
                action = trio.run(agent.buy_phase_action, player)
                yield
                action.apply(player)
                if player.discovered_cards:
                    discovered_card = trio.run(agent.discover_choice_action, player)
                    player.select_discover(discovered_card)

                if type(action) is EndPhaseAction:
                    break
            if len(player.in_play) > 1:
                arrangement = trio.run(agent.rearrange_cards, player)
                assert set(arrangement) == set(player.in_play)
                player.in_play = arrangement
        self.tavern.combat_step()
        if self.tavern.game_over():
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                trio.run(self.agents[name].game_over, player, position)

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
        trio.run(self._async_start_game)

    async def _async_start_game(self):
        player_choices = {}

        async def set_player_choice(player_name, player):
            player_choices[player_name] = await self.agents[player_name].hero_choice_action(player)

        async with trio.open_nursery() as nursery:
            for player_name, player in self.tavern.players.items():
                nursery.start_soon(set_player_choice, player_name, player)

        for player_name, player in self.tavern.players.items():
            player.choose_hero(player_choices[player_name])

    def play_round(self):
        return trio.run(self._async_play_round)

    async def _async_play_round(self):
        self.tavern.buying_step()

        async def perform_player_actions(agent, player):
            for _ in range(20):
                if player.discovered_cards:
                    discovered_card = await agent.discover_choice_action(player)
                    player.select_discover(discovered_card)
                else:
                    action = await agent.buy_phase_action(player)
                    action.apply(player)
                    if type(action) is EndPhaseAction:
                        break
            if len(player.in_play) > 1:
                arrangement = await agent.rearrange_cards(player)
                assert set(arrangement) == set(player.in_play)
                player.in_play = arrangement

        async with trio.open_nursery() as nursery:
            for player_name, player in self.tavern.players.items():
                if player.health <= 0:
                    continue
                nursery.start_soon(perform_player_actions, self.agents[player_name], player)

        self.tavern.combat_step()
        if self.tavern.game_over():
            async with trio.open_nursery() as nursery:
                for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                    nursery.start_soon(self.agents[name].game_over, player, position)

    def game_over(self):
        return self.tavern.game_over()

    def play_game(self):
        self.start_game()
        while not self.game_over():
            self.play_round()
