from typing import List, Tuple

from hearthstone.simulator.agent import Action, EndPhaseAction
from hearthstone.simulator.core.randomizer import DefaultRandomizer
from hearthstone.simulator.core.tavern import Tavern


class Replay:

    def __init__(self, seed: int, players: List[str], actions: List[Tuple[str, 'Action']]):
        self.seed = seed
        self.players = players
        self.actions = actions

    def run_replay(self) -> 'Tavern':
        randomizer = DefaultRandomizer(self.seed)
        tavern = Tavern()
        tavern.randomizer = randomizer
        for player in self.players:
            tavern.add_player(player)
        tavern.buying_step()
        end_phase_actions = set()
        for name, action in self.actions:
            if action is EndPhaseAction:
                assert name not in end_phase_actions
                end_phase_actions.add(name)
            action.apply(tavern.players[name])
            if len(end_phase_actions) == len(self.players):
                tavern.combat_step()
                end_phase_actions = set()
                tavern.buying_step()

