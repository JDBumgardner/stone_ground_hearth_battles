import typing
from typing import Dict
from hearthstone.tavern import Tavern
from hearthstone.agent import EndPhaseAction
if typing.TYPE_CHECKING:
    from hearthstone.agent import Agent


class RoundRobinHost:
    tavern: Tavern
    agents: Dict[str, 'Agent']

    def __init__(self, agents: Dict[str, 'Agent']):
        self.tavern = Tavern()
        self.agents = agents
        for player_name in agents.keys():
            self.tavern.add_player(player_name)

    async def start_game(self):
        for player_name, player in self.tavern.players.items():
            player.choose_hero(await self.agents[player_name].hero_choice_action(player))

    async def play_round_generator(self) -> typing.AsyncGenerator:  # TODO: think about how to test this code
        self.tavern.buying_step()
        for player_name, player in self.tavern.players.items():
            if player.health <= 0:
                continue
            agent = self.agents[player_name]
            for _ in range(20):
                action = await agent.buy_phase_action(player)
                yield
                action.apply(player)
                if player.discovered_cards:
                    discovered_card = await agent.discover_choice_action(player)
                    player.select_discover(discovered_card)

                if type(action) is EndPhaseAction:
                    break
            if len(player.in_play) > 1:
                arrangement = await agent.rearrange_cards(player)
                assert set(arrangement) == set(player.in_play)
                player.in_play = arrangement
        self.tavern.combat_step()
        if self.tavern.game_over():
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                self.agents[name].game_over(player, position)

    async def play_round(self):
        async for _ in self.play_round_generator():
            pass

    def game_over(self):
        return self.tavern.game_over()

    async def play_game(self):
        await self.start_game()
        while not self.game_over():
            await self.play_round()
