from typing import List

from hearthstone.simulator.agent import EndPhaseAction
from hearthstone.simulator.core.randomizer import DefaultRandomizer
from hearthstone.simulator.core.tavern import Tavern


class ReplayStep:
    def __init__(self, player: str, action: 'Action', bot_info=None):
        self.player = player
        self.action = action
        self.bot_info = bot_info


class Replay:
    def __init__(self, seed: int, players: List[str], replay_steps: List[ReplayStep]):
        self.seed = seed
        self.players = players
        self.steps :List[ReplayStep] = replay_steps

    def run_replay(self) -> 'Tavern':
        randomizer = DefaultRandomizer(self.seed)
        tavern = Tavern()
        tavern.randomizer = randomizer
        for player in self.players:
            tavern.add_player(player)
        tavern.buying_step()
        end_phase_actions = set()
        for replay_step in self.steps:
            if replay_step.action is EndPhaseAction:
                assert replay_step.name not in end_phase_actions
                end_phase_actions.add(replay_step.name)
            replay_step.action.apply(tavern.players[replay_step.name])
            if len(end_phase_actions) == len(self.players):
                tavern.combat_step()
                end_phase_actions = set()
                tavern.buying_step()

