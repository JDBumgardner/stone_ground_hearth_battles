import itertools
import typing
from typing import Dict, List, Optional

from hearthstone.asyncio import asyncio_utils
from hearthstone.simulator.agent.actions import EndPhaseAction
from hearthstone.simulator.agent.agent import AnnotatingAgent
from hearthstone.simulator.core.randomizer import Randomizer
from hearthstone.simulator.host.host import Host
from hearthstone.simulator.replay.observer import Observer
from hearthstone.simulator.replay.replay import Replay


class RoundRobinHost(Host):
    def __init__(self, agents: Dict[str, 'AnnotatingAgent'],
                 observers: Optional[List['Observer']] = None,
                 randomizer: Optional[Randomizer] = None):
        super().__init__(agents, observers, randomizer)

    def start_game(self):
        for player_name, player in self.tavern.players.items():
            hero_choice_action = asyncio_utils.get_or_create_event_loop().run_until_complete(
                self.agents[player_name].hero_choice_action(player))
            self._apply_and_record(player_name, hero_choice_action)

    def play_round_generator(self) -> typing.Generator:  # TODO: think about how to test this code
        self.tavern.buying_step()
        for player_name, player in self.tavern.players.items():
            agent = self.agents[player_name]
            for i in itertools.count():
                if player.dead:
                    break
                if player.discover_queue:
                    discover_choice_action, agent_annotation = asyncio_utils.get_or_create_event_loop().run_until_complete(
                        agent.annotated_discover_choice_action(player))
                    self._apply_and_record(player_name, discover_choice_action, agent_annotation)
                elif i > 40:
                    break
                else:
                    action, agent_annotation = asyncio_utils.get_or_create_event_loop().run_until_complete(
                        agent.annotated_buy_phase_action(player))
                    self._apply_and_record(player_name, action, agent_annotation)
                    yield
                    if type(action) is EndPhaseAction:
                        break
            if player.dead:
                continue
            if len(player.in_play) > 1:
                rearrange_action, agent_annotation = asyncio_utils.get_or_create_event_loop().run_until_complete(
                    agent.annotated_rearrange_cards(player))
                self._apply_and_record(player_name, rearrange_action, agent_annotation)
        self.tavern.combat_step()
        if self.tavern.game_over():
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                annotation = asyncio_utils.get_or_create_event_loop().run_until_complete(
                    self.agents[name].game_over(player, position))
                self.replay.agent_annotate(name, annotation)
            self._on_game_over()

    def play_round(self):
        for _ in self.play_round_generator():
            pass

    def game_over(self):
        return self.tavern.game_over()

    def play_game(self):
        self.start_game()
        while not self.game_over():
            self.play_round()

    def get_replay(self) -> Replay:
        return self.replay
