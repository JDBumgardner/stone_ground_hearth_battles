import typing
from typing import Dict, Optional, List

from frozenlist.frozen_list import FrozenList
from hearthstone.simulator import agent
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.simulator.replay.replay import Replay, ReplayStep

if typing.TYPE_CHECKING:
    from hearthstone.simulator.agent import Action, AnnotatingAgent
    from hearthstone.simulator.core.randomizer import Randomizer
    from hearthstone.simulator.observer import Observer


class Host:
    tavern: Tavern
    agents: Dict[str, 'AnnotatingAgent']
    replay: Replay
    observers: FrozenList # [Observer]

    def __init__(self, agents: Dict[str, 'AnnotatingAgent'], observers: Optional[List['Observer']] = None,
                 randomizer: Optional['Randomizer'] = None):
        self.tavern = Tavern()
        if randomizer:
            self.tavern.randomizer = randomizer
        self.agents = agents
        for player_name in sorted(agents.keys()):  # Sorting is important for replays to be exact with RNG.
            self.tavern.add_player(player_name)
        self.replay = Replay(self.tavern.randomizer.seed, list(self.tavern.players.keys()))
        if observers:
            self.observers = FrozenList(observers)

    def _apply_and_record(self, player_name: str, action: 'Action', agent_annotation: agent.Annotation = None):
        observer_annotations = {}
        for observer in self.observers:
            annotation = observer.on_action(self.tavern, player_name, action)
            if annotation is not None:
                observer_annotations[observer.name()] = annotation

        action.apply(self.tavern.players[player_name])
        self.replay.append_action(ReplayStep(player_name, action, agent_annotation, observer_annotations))

    def _on_game_over(self):
        for observer in self.observers:
            annotation = observer.on_game_over(self.tavern)
            if annotation is not None:
                self.replay.observer_annotate(observer.name(), annotation)

    def start_game(self):
        raise NotImplementedError()

    def play_round(self):
        raise NotImplementedError()

    def game_over(self):
        raise NotImplementedError()

    def play_game(self):
        raise NotImplementedError()

    def get_replay(self) -> Replay:
        raise NotImplementedError()


