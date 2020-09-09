from typing import Union, Tuple

from hearthstone.card_factory import make_metaclass
from hearthstone.events import BuyPhaseContext, CombatPhaseContext, CardEvent

VALHALLA = []

HeroType = make_metaclass(VALHALLA.append, ("Hero", "EmptyHero"))


class Hero(metaclass=HeroType):
    power_cost = 2
    hero_power_used = False
    can_use_power = True
    current_type = None
    buy_counter = 0
    digs_left = 5

    def __repr__(self):
        return str(type(self).__name__)

    def starting_health(self) -> int:
        return 40

    def minion_cost(self) -> int:
        return 3

    def refresh_cost(self) -> int:
        return 1

    def tavern_upgrade_costs(self) -> Tuple[int, int, int, int, int, int]:
        return (0, 5, 7, 8, 9, 10)

    def handle_event(self, event: CardEvent, context: Union[BuyPhaseContext, CombatPhaseContext]):
        pass

    def hero_power(self, context: BuyPhaseContext):
        assert self.hero_power_valid(context)
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
