import enum


class MONSTER_TYPES(enum.Enum):
    BEAST = 1
    MECH = 2
    PIRATE = 3
    DRAGON = 4
    DEMON = 5
    MURLOC = 6
    ELEMENTAL = 7
    ALL = 8

    @classmethod
    def single_types(cls):
        return [monster_type for monster_type in cls if monster_type != cls.ALL]
