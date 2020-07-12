from inspect import getmembers, isfunction

from hearthstone.battlebots.priority_functions import racist_priority_bot
from hearthstone.battlebots.hero_bot import HeroBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots import priority_functions
from hearthstone.monster_types import MONSTER_TYPES


def get_priority_bot_contestant_tuples():
    priority_bots = [PriorityBot, HeroBot]

    function_list = [member[1] for member in getmembers(priority_functions, isfunction)]
    contestant_tuples = []
    seed = 0
    for bot in priority_bots:
        for function in function_list:
            if function is not racist_priority_bot:
                seed += 1
                contestant_tuples.append((f"{bot.__name__}-{function.__name__}", function(seed, bot)))
            else:
                for monster_type in MONSTER_TYPES:
                    seed += 1
                    contestant_tuples.append((f"{bot.__name__}{function.__name__}{monster_type}",
                                        function(seed, bot, monster_type)))
    return contestant_tuples
