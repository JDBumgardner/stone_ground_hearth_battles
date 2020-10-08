import asyncio
import typing
from typing import Dict

from hearthstone.simulator import agent
from hearthstone.simulator.agent import EndPhaseAction, Action
from hearthstone.simulator.core.randomizer import Randomizer
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.simulator.host.host import Host
from hearthstone.simulator.replay.replay import Replay, ReplayStep


class RoundRobinHost(Host):
    tavern: Tavern
    agents: Dict[str, 'AnnotatingAgent']
    replay: Replay

    def __init__(self, agents: Dict[str, 'AnnotatingAgent'], randomizer: typing.Optional[Randomizer] = None):
        self.tavern = Tavern()
        if randomizer:
            self.tavern.randomizer = randomizer
        self.agents = agents
        for player_name in sorted(agents.keys()):  # Sorting is important for replays to be exact with RNG.
            self.tavern.add_player(player_name)
        self.replay = Replay(self.tavern.randomizer.seed, list(self.tavern.players.keys()))
        for player_name, player in self.tavern.players.items():
            hero_choice_action = asyncio.get_event_loop().run_until_complete(self.agents[player_name].hero_choice_action(player))
            self._apply_and_record(player_name, hero_choice_action)

    def _apply_and_record(self, player_name: str, action: Action, agent_annotation: agent.Annotation = None):
        action.apply(self.tavern.players[player_name])
        self.replay.append_action(ReplayStep(player_name, action, agent_annotation))

    def play_round_generator(self) -> typing.Generator:  # TODO: think about how to test this code
        self.tavern.buying_step()
        for player_name, player in self.tavern.players.items():
            if player.dead:
                continue
            agent = self.agents[player_name]
            for _ in range(40):
                action, agent_annotation = asyncio.get_event_loop().run_until_complete(agent.annotated_buy_phase_action(player))
                yield
                self._apply_and_record(player_name, action, agent_annotation)
                if player.discover_queue:
                    discover_choice_action = asyncio.get_event_loop().run_until_complete(agent.discover_choice_action(player))
                    self._apply_and_record(player_name, discover_choice_action)
                if type(action) is EndPhaseAction:
                    break
            if len(player.in_play) > 1:
                rearrange_action = asyncio.get_event_loop().run_until_complete(agent.rearrange_cards(player))
                self._apply_and_record(player_name, rearrange_action)
        self.tavern.combat_step()
        if self.tavern.game_over():
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                annotation = asyncio.get_event_loop().run_until_complete(self.agents[name].game_over(player, position))
                self.replay.annotate_replay(name, annotation)

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