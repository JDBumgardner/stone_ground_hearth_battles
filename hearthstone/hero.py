from typing import Union

from hearthstone.cards import CardEvent
from hearthstone.events import BuyPhaseContext, CombatPhaseContext


class Hero:
    power_cost = 2
    hero_power_used = False

    def starting_health(self) -> int:
        return 40

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        pass

    def hero_power(self, context: BuyPhaseContext):
        assert self.hero_power_valid()
        context.owner.coins -= self.power_cost
        self.hero_power_used = True
        self.hero_power_impl(context)

    def hero_power_impl(self, context: BuyPhaseContext):
        pass

    def hero_power_valid(self, context: BuyPhaseContext):
        if context.owner.coins < self.power_cost:
            return False
        if self.hero_power_used:
            return False
        if not self.hero_power_valid_impl(context):
            return False
        return True

    def hero_power_valid_impl(self, context: BuyPhaseContext):
        return True

    def on_buy_step(self):
        self.hero_power_used = False

class EmptyHero(Hero):
    pass
