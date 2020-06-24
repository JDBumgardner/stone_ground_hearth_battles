from typing import Union

from hearthstone.cards import CardEvent
from hearthstone.events import BuyPhaseContext, CombatPhaseContext


class Hero:
    def starting_health(self) -> int:
        return 40

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        pass

    def hero_power(self, context: BuyPhaseContext):
        pass


class EmptyHero(Hero):
    pass
