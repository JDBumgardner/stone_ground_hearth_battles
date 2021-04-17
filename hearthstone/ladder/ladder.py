import json
import random
from datetime import datetime
from typing import List, Callable

import trueskill

from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.get_bot_contestants import get_priority_bot_contestant_tuples, \
    get_priority_heuristics_bot_contestant_tuples
from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.battlebots.saurolisk_bot import SauroliskBot
from hearthstone.battlebots.supremacy_bot import SupremacyBot
from hearthstone.simulator.agent.agent import AnnotatingAgent
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.host.round_robin_host import RoundRobinHost
from hearthstone.training.pytorch.agents.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.encoding.default_encoder import DefaultEncoder
from hearthstone.training.pytorch.networks.save_load import load_from_saved


class ContestantAgentGenerator:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.function(*self.args, **self.kwargs)


class Contestant:
    def __init__(self, name, agent_generator: Callable[[], AnnotatingAgent], initial_trueskill=None):
        self.name = name
        self.agent_generator = agent_generator
        self.elo = 1200
        self.trueskill = initial_trueskill or trueskill.Rating()
        self.games_played = 0

    def __repr__(self):
        return f'(Agent "{self.name}" Trueskill {self.trueskill.mu:.2f})'


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

    new_trueskills = trueskill.rate([(contestant.trueskill,) for contestant in outcome], ranks=range(len(outcome)))
    for new_trueskill, contestant in zip(new_trueskills, outcome):
        contestant.trueskill = new_trueskill[0]


def print_standings(contestants: List[Contestant]):
    contestants = sorted(contestants, key=lambda c: c.trueskill.mu, reverse=True)
    print(contestants)


def run_tournament(contestants: List[Contestant], num_rounds=10, game_size=8):
    agents = {contestant.name: contestant.agent_generator() for contestant in contestants}
    for _ in range(num_rounds):
        round_contestants = random.sample(contestants, k=game_size)
        host = RoundRobinHost({c.name: agents[c.name] for c in round_contestants})
        host.play_game()
        winner_names = list(reversed([name for name, player in host.tavern.losers]))
        print(host.tavern.losers[-1][1].in_play, "-", host.tavern.losers[-1][1].hero, host.tavern.losers[-1][1].name)
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
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant(f"SupremacyUpgradeBot {t}", lambda: SupremacyBot(t, True, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant("SauroliskBot", lambda: SauroliskBot(5))]
    all_bots += [Contestant(name, lambda: bot) for name, bot in get_priority_bot_contestant_tuples()]
    all_bots += [Contestant(name, lambda: bot) for name, bot in get_priority_heuristics_bot_contestant_tuples()]
    return all_bots


def saved_learningbot_1v1_contestants() -> List[Contestant]:
    hparams = {
        "resume": False,
        'resume.from': '2020-10-18T02:14:22.530381',
        'export.enabled': True,
        'export.period_epochs': 200,
        'export.path': datetime.now().isoformat(),
        'opponents.initial': 'easiest',
        'opponents.self_play.enabled': True,
        'opponents.self_play.only_champions': True,
        'opponents.max_pool_size': 7,
        'adam.lr': 0.0001,
        'batch_size': 1024,
        'minibatch_size': 1024,
        'cuda': True,
        'entropy_weight': 0.001,
        'gae_gamma': 0.999,
        'gae_lambda': 0.9,
        'game_size': 2,
        'gradient_clipping': 0.5,
        'approx_kl_limit': 0.015,
        'nn.architecture': 'transformer',
        'nn.hidden_layers': 1,
        'nn.hidden_size': 32,
        'nn.activation': 'gelu',
        'nn.shared': False,
        'nn.encoding.redundant': True,
        'normalize_advantage': True,
        'normalize_observations': False,
        'num_workers': 1,
        'optimizer': 'adam',
        'policy_weight': 0.581166675499831,
        'ppo_epochs': 8,
        'ppo_epsilon': 0.1}

    all_bots = []
    # Jeremy has this bot, ask him for it!
    all_bots += [Contestant("LearningBot94200", lambda: PytorchBot(
        load_from_saved("../../data/learning/pytorch/saved_models/2020-10-30T20:50:44.311231/94200", hparams),
        DefaultEncoder(),
        annotate=False))]
    return all_bots


def load_ratings(contestants: List[Contestant], path):
    with open(path) as f:
        standings = json.load(f)
    standings_dict = dict(standings)
    for contestant in contestants:
        if contestant.name in standings_dict:
            contestant.elo = standings_dict[contestant.name]["elo"]
            contestant.trueskill = trueskill.Rating(standings_dict[contestant.name]["trueskill.mu"],
                                                    standings_dict[contestant.name]["trueskill.sigma"])
            contestant.games_played = standings_dict[contestant.name]["games_played"]


def save_ratings(contestants: List[Contestant], path):
    ranked_contestants = sorted(contestants, key=lambda c: c.trueskill, reverse=True)
    standings = [
        (c.name, {"elo": c.elo,
                  "trueskill.mu": c.trueskill.mu,
                  "trueskill.sigma": c.trueskill.sigma,
                  "games_played": c.games_played,
                  "last_time_updated": datetime.now().isoformat(),
                  "authors": c.agent_generator().authors}) for c
        in ranked_contestants]
    with open(path, "w") as f:
        json.dump(standings, f, indent=4)
