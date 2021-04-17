from typing import Any

from hearthstone.simulator.agent.actions import Action
from hearthstone.simulator.core.tavern import Tavern


Annotation = Any


class Observer:
    def name(self) -> str:
        pass

    def on_action(self, tavern: 'Tavern', player: str, action: 'Action') -> Annotation:
        pass

    def on_game_over(self, tavern: 'Tavern') -> Annotation:
        pass
