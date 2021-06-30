import enum


class MONSTER_TYPES(enum.Enum):
    BEAST = 1
    MECH = 2
    PIRATE = 3
    DRAGON = 4
    DEMON = 5
    MURLOC = 6
    ELEMENTAL = 7
    QUILBOAR = 8
    NEUTRAL = 9
    ALL = 10

    @classmethod
    def single_types(cls):
        return [monster_type for monster_type in cls if monster_type not in (cls.ALL, cls.NEUTRAL)]
