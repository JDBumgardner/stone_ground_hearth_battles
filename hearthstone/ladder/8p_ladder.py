from hearthstone.ladder.ladder import all_contestants, run_tournament, load_ratings, save_ratings, \
    saved_learningbot_1v1_contestants


def main():
    contestants = all_contestants()
    standings_path = "../../data/standings/8p.json"
    load_ratings(contestants, standings_path)
    run_tournament(contestants, 1000, 8)
    save_ratings(contestants, standings_path)


if __name__ == "__main__":
    main()
