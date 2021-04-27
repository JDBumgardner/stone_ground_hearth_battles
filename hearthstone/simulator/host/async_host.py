import asyncio
import itertools
from typing import Dict, Optional, List

import logging

from hearthstone.asyncio import asyncio_utils
from hearthstone.simulator.agent.actions import EndPhaseAction
from hearthstone.simulator.core.randomizer import Randomizer
from hearthstone.simulator.host.host import Host
from hearthstone.simulator.replay.replay import Replay

logger = logging.getLogger(__name__)

class AsyncHost(Host):

    def start_game(self):
        asyncio_utils.get_or_create_event_loop().run_until_complete(self.async_start_game())

    async def async_start_game(self):
        async def set_player_choice(player_name, player):
            hero_choice = await self.agents[player_name].hero_choice_action(player)
            self._apply_and_record(player_name, hero_choice)

        player_choice_tasks = []
        for player_name, player in self.tavern.players.items():
            player_choice_tasks.append(asyncio_utils.create_task(set_player_choice(player_name, player), logger=logger))
        await asyncio.gather(*player_choice_tasks)

    def play_round(self):
        return asyncio_utils.get_or_create_event_loop().run_until_complete(self.async_play_round())

    async def async_play_round(self):
        self.tavern.buying_step()

        async def perform_player_actions(player_name, agent, player):
            for i in itertools.count():
                if player.dead:
                    return
                if player.discover_queue:
                    discover_card_action, agent_annotation = await agent.annotated_discover_choice_action(player)
                    self._apply_and_record(player_name, discover_card_action, agent_annotation)
                elif player.hero.discover_choices:
                    hero_discover_action, agent_annotation = await agent.annotated_hero_discover_action(player)
                    self._apply_and_record(player_name, hero_discover_action, agent_annotation)
                elif i > 40:
                    break
                else:
                    action, agent_annotation = await agent.annotated_buy_phase_action(player)
                    self._apply_and_record(player_name, action, agent_annotation)
                    if type(action) is EndPhaseAction:
                        break
            if len(player.in_play) > 1:
                rearrange_action, agent_annotation = await agent.annotated_rearrange_cards(player)
                self._apply_and_record(player_name, rearrange_action, agent_annotation)

        perform_player_action_tasks = []
        for player_name, player in self.tavern.players.items():
            if player.dead:
                continue
            perform_player_action_tasks.append(
                asyncio_utils.create_task(perform_player_actions(player_name, self.agents[player_name], player), logger=logger))
        await asyncio.gather(*perform_player_action_tasks)

        self.tavern.combat_step()
        if self.tavern.game_over():
            async def report_game_over(name, player, position):
                annotation = await self.agents[name].game_over(player, position)
                self.replay.agent_annotate(name, annotation)

            game_over_tasks = []
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                game_over_tasks.append(asyncio_utils.create_task(report_game_over(name, player, position), logger=logger))
            await asyncio.gather(*game_over_tasks)
            self._on_game_over()

    def game_over(self):
        return self.tavern.game_over()

    async def async_play_game(self):
        await self.async_start_game()
        while not self.game_over():
            await self.async_play_round()

    def play_game(self):
        return asyncio_utils.get_or_create_event_loop().run_until_complete(self.async_play_game())

    def get_replay(self) -> Replay:
        return self.replay
