from typing import Tuple

from hearthstone.simulator.core.hero import Hero


class Bartendotron(Hero):
    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return (0, 4, 6, 7, 8, 9)
