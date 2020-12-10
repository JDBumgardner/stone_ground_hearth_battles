import enum
import typing

if typing.TYPE_CHECKING:
    from hearthstone.simulator.core.hero import Hero


class SECRETS(enum.Enum):
    ICE_BLOCK = 1
    SPLITTING_IMAGE = 2
    AUTODEFENSE_MATRIX = 3
    VENOMSTRIKE_TRAP = 4
    REDEMPTION = 5
    AVENGE = 6
    SNAKE_TRAP = 7

    @classmethod
    def remaining_secrets(cls, hero: 'Hero'):
        return [secret for secret in cls if secret not in hero.secrets]