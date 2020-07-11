from inspect import getmembers, isfunction

from hearthstone.battlebots.priority_functions import racist_priority_bot
from hearthstone.battlebots.hero_bot import HeroBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots import priority_functions
from hearthstone.monster_types import MURLOC, BEAST, DRAGON, DEMON, PIRATE, MECH


def get_priority_bot_contestant_tuples():
    priority_bots = [PriorityBot, HeroBot]
    # # TODO: JACK!! The below array makes me sad, how should we do this?  Thanks, Adam
    # priority_functions = [attack_health_priority_bot, attack_health_tripler_priority_bot, racist_priority_bot,
    #                       priority_saurolisk_bot, priority_saurolisk_buff_bot, priority_adaptive_tripler_bot,
    #                       priority_health_tripler_bot, priority_attack_tripler_bot, battlerattler_priority_bot,
    #                       priority_pogo_hopper_bot]
    function_list = [member[1] for member in getmembers(priority_functions, isfunction)]
    contestant_tuples = []
    monster_types = [MURLOC, BEAST, MECH, DRAGON, DEMON, PIRATE]
    seed = 0
    for bot in priority_bots:
        for function in function_list:
            if function is not racist_priority_bot:
                seed += 1
                contestant_tuples.append((f"{bot.__name__}-{function.__name__}", function(seed, bot)))
            else:
                for monster_type in monster_types:
                    seed += 1
                    contestant_tuples.append((f"{bot.__name__}{function.__name__}{monster_type}",
                                        function(seed, bot, monster_type)))
    return contestant_tuples
