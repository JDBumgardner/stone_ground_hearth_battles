from inspect import getmembers, isfunction

from hearthstone.battlebots.CardSpecificHeuristics import MamasLove, SameTypeAdvantage, DragonPayoffs, \
    MonstrousMacawPower
from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.battlebots.hero_bot import HeroBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.simulator.core.monster_types import MONSTER_TYPES


def get_priority_bot_contestant_tuples():
    priority_bots = [PriorityBot, HeroBot, EarlyGameBot]

    function_list = [member[1] for member in getmembers(PriorityFunctions, isfunction)]
    contestant_tuples = []
    seed = 0
    for bot in priority_bots:
        for function in function_list:
            if function is PriorityFunctions.racist_priority_bot:
                for monster_type in MONSTER_TYPES:
                    if monster_type != MONSTER_TYPES.ALL:
                        seed += 1
                        contestant_tuples.append((f"{bot.__name__}-{function.__name__}-{monster_type.name}",
                                            function(seed, bot, monster_type)))
            elif function is not PriorityFunctions.priority_callables_bot:
                seed += 1
                contestant_tuples.append((f"{bot.__name__}-{function.__name__}", function(seed, bot)))
    return contestant_tuples



def get_priority_heuristics_bot_contestant_tuples():
    priority_bots = [PriorityBot, HeroBot, EarlyGameBot]

    function_list = [member[1] for member in getmembers(PriorityFunctions, isfunction)]

    contestant_tuples = []
    seed = 0
    for bot in priority_bots:
        for function in function_list:
            if function is PriorityFunctions.priority_callables_bot:
                contestant_tuples.append((f"{bot.__name__}-{function.__name__}", function(seed, bot, None, [MamasLove(), SameTypeAdvantage(),
                                                                                                            DragonPayoffs(), MonstrousMacawPower()])))
                seed +=1
    return contestant_tuples