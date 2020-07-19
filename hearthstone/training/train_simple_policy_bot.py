import logging
import random

from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.priority_functions import racist_priority_bot
from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.battlebots.simple_policy_bot import SimplePolicyBot
from hearthstone.battlebots.supremacy_bot import SupremacyBot
from hearthstone.host import RoundRobinHost
from hearthstone.ladder.ladder import Contestant, update_ratings, print_standings, save_ratings
from hearthstone.monster_types import MONSTER_TYPES.DRAGON, MONSTER_TYPES.MECH, MONSTER_TYPES.BEAST, MONSTER_TYPES.MURLOC, MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE

learning_rate = .1


def learning_bot_opponents():
    all_bots = [Contestant(f"RandomBot_{i}", RandomBot(1)) for i in range(20)]
    all_bots += [Contestant(f"CheapoBot", CheapoBot(3))]
    all_bots += [Contestant(f"SupremacyBot {t}", SupremacyBot(t, False, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON, MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant(f"PriorityRacistBot {t}", racist_priority_bot(t, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON, MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    return all_bots


def main():
    logging.getLogger().setLevel(logging.INFO)
    other_contestants = learning_bot_opponents()
    learning_bot = SimplePolicyBot(None, 1)
    learning_bot_contestant = Contestant("LearningBot", learning_bot)
    contestants = other_contestants + [learning_bot_contestant]
    bot_file = "../../data/learning/simple_policy_bot.1.json"
    standings_path = "../../data/learning/standings_policy_bot.json"
    # learning_bot.read_from_file(bot_file)
    # load_ratings(contestants, standings_path)

    for _ in range(10000):
        round_contestants = [learning_bot_contestant] + random.sample(other_contestants, k=7)
        host = RoundRobinHost({contestant.name: contestant.agent_generator() for contestant in round_contestants})
        host.play_game()
        winner_names = list(reversed([name for name, player in host.tavern.losers]))
        print("---------------------------------------------------------------")
        print(winner_names)
        #print(host.tavern.losers[-1][1].in_play)
        ranked_contestants = sorted(round_contestants, key=lambda c: winner_names.index(c.name))
        update_ratings(ranked_contestants)
        print_standings(contestants)
        for contestant in round_contestants:
            contestant.games_played += 1
        if learning_bot_contestant in round_contestants:
            learning_bot.learn_from_game(ranked_contestants.index(learning_bot_contestant), learning_rate)
            print("Favorite cards: ", sorted(learning_bot.priority_buy_dict.items(), key=lambda item: item[1], reverse=True))
            # learning_bot.save_to_file(bot_file)

    save_ratings(contestants, standings_path)


if __name__ == '__main__':
    main()
