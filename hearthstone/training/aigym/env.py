import math

import gym
from gym import spaces

# TODO: Derive (some of) these constants from game data
MAX_HEALTH = 50
NUM_HEROES = 41
HAND_SIZE = 10
BOARD_SIZE = 7
NUM_PLAYERS = 8
NUM_MINIONS = 108
NUM_TOKENS = 10
# These can probably actually be unbounded...
MAX_ATTACK = 10000
MAX_DEFENSE = 10000
NUM_MODIFIERS = 10
NUM_TRIBES = 7
NUM_DISCOVERS = 5
MAX_TURNS = 50
NUM_TIERS = 6
MAX_LEVEL_COST = 11
MAX_GOLD = 10
MAX_DEATHRATTLES = 3


def CountActions():
    play_card = HAND_SIZE * (BOARD_SIZE) * (BOARD_SIZE - 1)
    use_hero_power = BOARD_SIZE + 1
    refresh = 1
    buy_card = BOARD_SIZE
    tier_up = 1
    play_discover = 1
    end_turn = 1
    rearrange = math.factorial(BOARD_SIZE)
    total_actions = (play_card + use_hero_power + refresh + buy_card + tier_up + play_discover + end_turn + rearrange)
    return total_actions * NUM_HEROES


def card():
    # Add one for empty card
    card_type = spaces.Discrete(NUM_MINIONS + NUM_TOKENS + NUM_DISCOVERS + 1)
    is_golden = spaces.Discrete(2)
    attack_range = spaces.Discrete(MAX_ATTACK + 1)
    defense_range = spaces.Discrete(MAX_DEFENSE + 1)
    monster_type = spaces.Discrete(NUM_TRIBES + 1)
    modifiers = spaces.MultiBinary(NUM_MODIFIERS + 1)
    extra_deathrattles = spaces.MultiDiscrete(MAX_DEATHRATTLES)
    return spaces.Tuple(card_type, is_golden, attack_range, defense_range, monster_type, modifiers, extra_deathrattles)


class BattlegroundsEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(CountActions())
        self.observation_space = spaces.Dict()
        board_space = spaces.Dict()
        board_space["board_size"] = spaces.Discrete(BOARD_SIZE + 1)
        for i in range(BOARD_SIZE):
            board_space["board_" + i] = card()
        self.observation_space["board"] = board_space
        hero_space = spaces.Dict()
        hero_space["my_hero"] = spaces.Discrete(NUM_HEROES)
        hero_space["my_hero_health"] = spaces.Discrete(MAX_HEALTH + 1)
        hero_space["playing_next"] = spaces.Discrete(NUM_PLAYERS)
        for i in range(1, NUM_PLAYERS):
            hero_space["other_hero_" + i] = spaces.Discrete(NUM_HEROES)
            hero_space["other_hero_health_" + i] = spaces.Discrete(MAX_HEALTH + 1)
            hero_space["other_hero_streak_" + i] = spaces.Discrete(MAX_TURNS + 1)
            # This could be infinite?
            hero_space["other_hero_num_triples_" + i] = spaces.Discrete(MAX_TURNS * 4)
            hero_space["other_hero_played_0_" + i] = spaces.Discrete(NUM_PLAYERS)
            hero_space["other_hero_played_0_" + i] = spaces.Discrete(MAX_HEALTH * 2 + 1)
            hero_space["other_hero_damage_1_" + i] = spaces.Discrete(NUM_PLAYERS)
            hero_space["other_hero_damage_1_" + i] = spaces.Discrete(MAX_HEALTH * 2 + 1)
            hero_space["other_hero_tribe" + i] = spaces.Discrete(NUM_TRIBES + 1)
            hero_space["other_hero_tribe_count" + i] = spaces.Discrete(BOARD_SIZE + 1)
            hero_space["other_hero_prev_board" + i] = board_space.copy()
            hero_space["other_hero_last_seen" + i] = spaces.Discrete(MAX_TURNS + 1)
        self.observation_space["hero_info"] = hero_space
        hand_space = spaces.Dict()
        hand_space["hand_size"] = spaces.Discrete(HAND_SIZE + 1)
        for i in range(HAND_SIZE):
            hand_space["hand_" + i] = card()
        self.observation_space["hand"] = hand_space
        self.observation_space["tier"] = spaces.Discrete(NUM_TIERS)
        self.observation_space["level_cost"] = spaces.Discrete(MAX_LEVEL_COST + 1)
        self.observation_space["gold"] = spaces.MultiDiscrete([MAX_GOLD + 1, MAX_GOLD])
        self.power_cost["power_cost"] = spaces.Discrete(MAX_GOLD + 1)
        self.observation_space["power_available"] = spaces.Discrete(2)
        self.observation_space["refresh_cost"] = spaces.Discrete(MAX_GOLD + 1)
