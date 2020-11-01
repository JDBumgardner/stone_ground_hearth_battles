from hearthstone.ladder.ladder import all_contestants, load_ratings, run_tournament, save_ratings, \
    saved_learningbot_1v1_contestants


def main():
    contestants = all_contestants() + saved_learningbot_1v1_contestants()
    standings_path = "../../data/standings/1v1.json"
    load_ratings(contestants, standings_path)
    run_tournament(contestants, 1000, 2)
    save_ratings(contestants, standings_path)


if __name__ == "__main__":
    main()
