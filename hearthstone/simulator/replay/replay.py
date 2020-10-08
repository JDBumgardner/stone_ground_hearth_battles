from typing import List, Any

from hearthstone.simulator.agent import EndPhaseAction, Action, HeroChoiceAction
from hearthstone.simulator.core.randomizer import DefaultRandomizer
from hearthstone.simulator.core.tavern import Tavern


class ReplayStep:
    def __init__(self, player: str, action: 'Action', agent_annotation: Any = None):
        self.player = player
        self.action = action
        self.agent_annotation = agent_annotation

    def __repr__(self):
        return f"{self.player}: {self.action} ({self.agent_annotation})"

class Replay:
    def __init__(self, seed: int, players: List[str]):
        self.seed = seed
        self.players = players
        self.steps: List[ReplayStep] = []
        self.agent_annotations = {player: None for player in players}

    def append_action(self, replay_step: ReplayStep):
        self.steps.append(replay_step)

    def annotate_replay(self, player: str, annotation: Any):
        self.agent_annotations[player] = annotation

    def run_replay(self) -> 'Tavern':
        randomizer = DefaultRandomizer(self.seed)
        tavern = Tavern()
        tavern.randomizer = randomizer
        for player in sorted(self.players):  # Sorting is important for replays to be exact with RNG.
            tavern.add_player(player)

        hero_chosen_players = set()
        for replay_step in self.steps[:len(self.players)]:
            assert isinstance(replay_step.action, HeroChoiceAction)
            replay_step.action.apply(tavern.players[replay_step.player])
            hero_chosen_players.add(replay_step.player)
        assert hero_chosen_players == set(self.players)

        tavern.buying_step()
        end_phase_actions = set()
        for replay_step in self.steps[len(self.players):]:
            if replay_step.action is EndPhaseAction:
                assert replay_step.player not in end_phase_actions
                end_phase_actions.add(replay_step.player)
            replay_step.action.apply(tavern.players[replay_step.player])
            if len(end_phase_actions) == len(self.players):
                tavern.combat_step()
                end_phase_actions = set()
                tavern.buying_step()
        return tavern

