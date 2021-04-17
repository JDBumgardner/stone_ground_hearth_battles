from typing import List, Dict

from hearthstone.simulator.agent.actions import Action
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.simulator.replay.observer import Observer, Annotation


class FinalBoardAnnotator(Observer):
    """
    This annotator records the final boards for all players.
    """
    def name(self) -> str:
        return "FinalBoardAnnotator"

    def on_action(self, tavern: 'Tavern', player: str, action: 'Action') -> Annotation:
        return None

    def on_game_over(self, tavern: 'Tavern') -> Dict[str, tuple]:
        return {name: [str(card) for card in player.in_play] for name, player in tavern.players.items()}
