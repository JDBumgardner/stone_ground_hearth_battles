import enum

from hearthstone.simulator.core.player import Player


class SECRETS(enum.Enum):
    ICE_BLOCK = 1
    SPLITTING_IMAGE = 2
    AUTODEFENSE_MATRIX = 3
    VENOMSTRIKE_TRAP = 4
    EFFIGY = 5
    REDEMPTION = 6
    AVENGE = 7
    SNAKE_TRAP = 8

    @classmethod
    def remaining_secrets(cls, player: 'Player'):
        return [secret for secret in cls if secret not in player.hero.secrets]