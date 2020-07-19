import json
import random
from datetime import datetime
from typing import List, Callable

from hearthstone.agent import Agent
from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.get_bot_contestants import get_priority_bot_contestant_tuples
from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.battlebots.saurolisk_bot import SauroliskBot
from hearthstone.battlebots.supremacy_bot import SupremacyBot
from hearthstone.host import RoundRobinHost
from hearthstone.monster_types import MONSTER_TYPES

class Contestant:
    def __init__(self, name,  agent_generator: Callable[[], Agent]):
        self.name = name
        self.agent_generator = agent_generator
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


def run_tournament(contestants: List[Contestant], num_rounds=10):
    agents = {contestant.name: contestant.agent_generator() for contestant in contestants}
    for _ in range(num_rounds):
        round_contestants = random.sample(contestants, k=8)
        host = RoundRobinHost({c.name: agents[c.name] for c in round_contestants})
        host.play_game()
        winner_names = list(reversed([name for name, player in host.tavern.losers]))
        print(host.tavern.losers[-1][1].in_play)
        ranked_contestants = sorted(round_contestants, key=lambda c: winner_names.index(c.name))
        update_ratings(ranked_contestants)
        print_standings(contestants)
        for contestant in round_contestants:
            contestant.games_played += 1


def all_contestants():
    all_bots = [Contestant(f"RandomBot", lambda: RandomBot(1))]
    all_bots += [Contestant(f"NoActionBot ", lambda: NoActionBot())]
    all_bots += [Contestant(f"CheapoBot", lambda: CheapoBot(3))]
    all_bots += [Contestant(f"SupremacyBot {t}", lambda: SupremacyBot(t, False, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON, MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant(f"SupremacyUpgradeBot {t}", lambda: SupremacyBot(t, True, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON, MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant("SauroliskBot", lambda: SauroliskBot(5))]
    # all_bots += [Contestant("PriorityHealthAttackBot", attack_health_priority_bot(6))]
    # all_bots += [Contestant(f"PriorityRacistBot {t}", racist_priority_bot(t, i)) for i, t in
    #              enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON, MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    # all_bots += [Contestant("PrioritySauroliskBot", priority_saurolisk_bot(8))]
    # # all_bots += [Contestant("PriorityHealthAttackTriplerBot", attack_health_tripler_priority_bot(9))]
    # all_bots += [Contestant("PriorityAdaptiveTriplerBot", priority_adaptive_tripler_bot(10))]
    # all_bots += [Contestant("PriorityHeathTriplerBot", priority_health_tripler_bot(11))]
    # all_bots += [Contestant("PriorityAttackTriplerBot", priority_attack_tripler_bot(12))]
    # all_bots += [Contestant("BattleRattlerPriorityBot", battlerattler_priority_bot(13))]
    # all_bots += [Contestant("PogoHopperPriorityBot", priority_pogo_hopper_bot(14))]
    # learned_bot_1 = LearnedPriorityBot(None, 0, 15)
    # learned_bot_1.read_from_file("../../data/learning/priority_bot.1.json")
    # all_bots += [Contestant("LearnedPriorityBot1", learned_bot_1)]
    # all_bots += [Contestant("priority_st_ad_tr_bot", priority_st_ad_tr_bot(16))]
    # all_bots += [Contestant("PriorityBuffSauroliskBot", priority_saurolisk_buff_bot(17))]
    # all_bots += [Contestant("PriorityBuffSauroliskBot", priority_saurolisk_buff_bot(18))]
    #
    all_bots += [Contestant(name, lambda: bot) for name, bot in get_priority_bot_contestant_tuples()]
    return all_bots


def load_ratings(contestants: List[Contestant], path):
    # TODO: This is a hack for saving to a specific file.
    with open(path) as f:
        standings = json.load(f)
    standings_dict = dict(standings)
    for contestant in contestants:
        if contestant.name in standings_dict:
            contestant.elo = standings_dict[contestant.name]["elo"]
            contestant.games_played = standings_dict[contestant.name]["games_played"]


def save_ratings(contestants: List[Contestant], path):
    ranked_contestants = sorted(contestants, key=lambda c: c.elo, reverse=True)
    standings = [
        (c.name, {"elo": c.elo,
                  "games_played": c.games_played,
                  "last_time_updated": datetime.now().isoformat(),
                  "authors": c.agent_generator().authors}) for c
        in ranked_contestants]
    with open(path, "w") as f:
        json.dump(standings, f, indent=4)


def main():
    contestants = all_contestants()
    standings_path = "../../data/standings.json"
    load_ratings(contestants, standings_path)
    run_tournament(contestants, 100)
    save_ratings(contestants, standings_path)


if __name__ == "__main__":
    main()
