import json
import random
from datetime import datetime
from typing import List

from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.priority_bot import attack_health_priority_bot, racist_priority_bot, priority_saurolisk_bot, \
    attack_health_tripler_priority_bot, priority_adaptive_tripler_bot, priority_health_tripler_bot, \
    priority_attack_tripler_bot, battlerattler_priority_bot, priority_pogo_hopper_bot
from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.battlebots.saurolisk_bot import SauroliskBot
from hearthstone.battlebots.supremacy_bot import SupremacyBot
from hearthstone.host import RoundRobinHost
from hearthstone.monster_types import *


class Contestant:
    def __init__(self, name, agent):
        self.name = name
        self.agent = agent
        self.elo = 1200
        self.games_played = 0

    def __repr__(self):
        return f'(Agent "{self.name}" Elo {int(self.elo)})'


def probability_of_win(elo1: int, elo2: int) -> float:
    return 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400.0))


def update_ratings(outcome: List[Contestant]):
    coefficient: float = 30
    num_contestants = len(outcome)
    elo_delta = [0 for _ in range(num_contestants)]
    for i in range(num_contestants):
        for j in range(i, num_contestants):
            win_prob = probability_of_win(outcome[i].elo, outcome[j].elo)
            elo_delta[i] += coefficient * (1 - win_prob)
            elo_delta[j] += coefficient * (win_prob - 1)
    for contestant, elo_diff in zip(outcome, elo_delta):
        contestant.elo += elo_diff


def print_standings(contestants: List[Contestant]):
    contestants = sorted(contestants, key=lambda c: c.elo, reverse=True)
    print(contestants)


def run_tournament(contestants: List[Contestant], num_rounds = 10):
    for _ in range(num_rounds):
        round_contestants = random.sample(contestants, k=8)
        host = RoundRobinHost({contestant.name: contestant.agent for contestant in round_contestants})
        host.play_game()
        winner_names = list(reversed([name for name, player in host.tavern.losers]))
        print(host.tavern.losers[-1][1].in_play)
        ranked_contestants = sorted(round_contestants, key=lambda c: winner_names.index(c.name))
        update_ratings(ranked_contestants)
        print_standings(contestants)
        for contestant in round_contestants:
            contestant.games_played += 1


def all_contestants():
    all_bots = [Contestant(f"RandomBot", RandomBot(1))]
    all_bots += [Contestant(f"NoActionBot ", NoActionBot())]
    all_bots += [Contestant(f"CheapoBot", CheapoBot(3))]
    all_bots += [Contestant(f"SupremacyBot {t}", SupremacyBot(t, False, i)) for i, t in
                 enumerate([MURLOC, BEAST, MECH, DRAGON, DEMON, PIRATE])]
    all_bots += [Contestant(f"SupremacyUpgradeBot {t}", SupremacyBot(t, True, i)) for i, t in
                 enumerate([MURLOC, BEAST, MECH, DRAGON, DEMON, PIRATE])]
    all_bots += [Contestant("SauroliskBot", SauroliskBot(5))]
    all_bots += [Contestant("PriorityHealthAttackBot", attack_health_priority_bot(6))]
    all_bots += [Contestant(f"PriorityRacistBot {t}", racist_priority_bot(t, i)) for i, t in
                 enumerate([MURLOC, BEAST, MECH, DRAGON, DEMON, PIRATE])]
    all_bots += [Contestant("PrioritySauroliskBot", priority_saurolisk_bot(8))]
    all_bots += [Contestant("PriorityHealthAttackTriplerBot", attack_health_tripler_priority_bot(9))]
    all_bots += [Contestant("PriorityAdaptiveTriplerBot", priority_adaptive_tripler_bot(10))]
    all_bots += [Contestant("PriorityHeathTriplerBot", priority_health_tripler_bot(11))]
    all_bots += [Contestant("PriorityAttackTriplerBot", priority_attack_tripler_bot(12))]
    all_bots += [Contestant("BattleRattlerPriorityBot", battlerattler_priority_bot(13))]
    all_bots += [Contestant("PogoHopperPriorityBot", priority_pogo_hopper_bot(14))]
    return all_bots


def load_ratings(contestants: List[Contestant]):
    # TODO: This is a hack for saving to a specific file.
    with open("../../data/standings.json") as f:
        standings = json.load(f)
    standings_dict = dict(standings)
    for contestant in contestants:
        if contestant.name in standings_dict:
            contestant.elo = standings_dict[contestant.name]["elo"]
            contestant.games_played = standings_dict[contestant.name]["games_played"]


def save_ratings(contestants: List[Contestant]):
    ranked_contestants = sorted(contestants, key=lambda c: c.elo, reverse=True)
    standings = [
        (c.name, {"elo": c.elo,
                  "games_played": c.games_played,
                  "last_time_updated": datetime.now().isoformat(),
                  "authors": c.agent.authors}) for c
        in ranked_contestants]
    with open("../../data/standings.json", "w") as f:
        json.dump(standings, f, indent=4)


def main():
    contestants = all_contestants()
    load_ratings(contestants)
    run_tournament(contestants, 100)
    save_ratings(contestants)


if __name__ == "__main__":
    main()
