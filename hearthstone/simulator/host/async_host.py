import asyncio
from typing import Dict, Optional

from hearthstone.simulator import agent
from hearthstone.simulator.agent import EndPhaseAction, Action, AnnotatingAgent, BuyAction
from hearthstone.simulator.core.randomizer import Randomizer
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.simulator.host.host import Host
from hearthstone.simulator.replay.replay import Replay, ReplayStep


class AsyncHost(Host):
    tavern: Tavern
    agents: Dict[str, 'AnnotatingAgent']
    replay: Replay

    def __init__(self, agents: Dict[str, 'AnnotatingAgent'], randomizer: Optional[Randomizer]=None):
        self.tavern = Tavern()
        if randomizer:
            self.tavern.randomizer = randomizer
        self.agents = agents
        for player_name in sorted(agents.keys()): # Sorting is important for replays to be exact with RNG.
            self.tavern.add_player(player_name)
        self.replay = Replay(self.tavern.randomizer.seed, list(self.tavern.players.keys()))

    def _apply_and_record(self, player_name: str, action: Action, agent_annotation: agent.Annotation = None):
        action.apply(self.tavern.players[player_name])
        self.replay.append_action(ReplayStep(player_name, action, agent_annotation))

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
            self._apply_and_record(player_name, player_choices[player_name])

    def play_round(self):
        return asyncio.get_event_loop().run_until_complete(self._async_play_round())

    async def _async_play_round(self):
        self.tavern.buying_step()

        async def perform_player_actions(player_name, agent, player):
            for _ in range(40):
                if player.discover_queue:
                    discover_card_action = await agent.discover_choice_action(player)
                    self._apply_and_record(player_name, discover_card_action)
                else:
                    action, agent_annotation = await agent.annotated_buy_phase_action(player)
                    self._apply_and_record(player_name, action, agent_annotation)
                    if type(action) is EndPhaseAction:
                        break
            if len(player.in_play) > 1:
                rearrange_action = await agent.rearrange_cards(player)
                self._apply_and_record(player_name, rearrange_action)

        perform_player_action_tasks = []
        for player_name, player in self.tavern.players.items():
            if player.dead:
                continue
            perform_player_action_tasks.append(
                asyncio.create_task(perform_player_actions(player_name, self.agents[player_name], player)))
        await asyncio.gather(*perform_player_action_tasks)

        self.tavern.combat_step()
        if self.tavern.game_over():
            async def report_game_over(name, player):
                annotation = self.agents[name].game_over(player, position)
                self.replay.annotate_replay(name, annotation)
            game_over_tasks = []
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                game_over_tasks.append(asyncio.create_task(report_game_over(name, player)))
            await asyncio.gather(*game_over_tasks)

    def game_over(self):
        return self.tavern.game_over()

    def play_game(self):
        self.start_game()
        while not self.game_over():
            self.play_round()

    def get_replay(self) -> Replay:
        return self.replay