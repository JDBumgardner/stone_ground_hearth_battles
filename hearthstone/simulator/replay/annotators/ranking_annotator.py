from typing import List

from hearthstone.simulator.replay.observer import Observer, Annotation


class RankingAnnotator(Observer):
    """
    This annotator simply records the final ranking at the end of the game.
    """
    def name(self) -> str:
        return "RankingAnnotator"

    def on_action(self, tavern: 'Tavern', player: str, action: 'Action') -> Annotation:
        return None

    def on_game_over(self, tavern: 'Tavern') -> List[str]:
        return list(reversed([name for name, player in tavern.losers]))
