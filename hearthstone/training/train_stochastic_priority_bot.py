import logging
import random

from hearthstone.battlebots.stochastic_priority_bot import LearnedPriorityBot
from hearthstone.host import RoundRobinHost
from hearthstone.ladder.ladder import Contestant, update_ratings, print_standings, load_ratings, save_ratings, \
    all_contestants


def main():
    logging.getLogger().setLevel(logging.INFO)
    other_contestants = all_contestants()
    learning_bot = LearnedPriorityBot(None, 0.05, 10)
    learning_bot_contestant = Contestant("LearningBot", learning_bot)
    contestants = other_contestants + [learning_bot_contestant]
    bot_file = "../../data/learning/priority_bot.1.json"
    standings_path = "../../data/learning/standings.json"
    learning_bot.read_from_file(bot_file)
    load_ratings(contestants, standings_path)

    for _ in range(1000):
        round_contestants = [learning_bot_contestant] + random.sample(other_contestants, k=7)
        host = RoundRobinHost({contestant.name: contestant.agent for contestant in round_contestants})
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
            learning_bot.learn_from_game(ranked_contestants.index(learning_bot_contestant))
            print("Favorite cards: ", sorted(learning_bot.priority_dict.items(), key=lambda item: item[1], reverse=True))
            learning_bot.save_to_file(bot_file)

    save_ratings(contestants, standings_path)


if __name__ == '__main__':
    main()
