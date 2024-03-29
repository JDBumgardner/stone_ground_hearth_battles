import collections
import enum
from typing import Dict, Optional

import logging

from hearthstone.simulator.core import combat, events
from hearthstone.simulator.core.card_pool import PrintingPress
from hearthstone.simulator.core.cards import CardList
from hearthstone.simulator.core.combat import WarParty
from hearthstone.simulator.core.hero import Hero
from hearthstone.simulator.core.hero_pool import VALHALLA
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.player import Player
from hearthstone.simulator.core.randomizer import DefaultRandomizer, Randomizer

logger = logging.getLogger(__name__)

class Tavern:
    def __init__(self, randomizer: Optional[Randomizer] = None,
                 restrict_types: Optional[bool] = True, include_graveyard: Optional[bool] = False):
        self.players: collections.OrderedDict[str, Player] = collections.OrderedDict()
        self.turn_count = 0
        self._max_turn_count = 50
        self.current_player_pairings = []
        self.randomizer = randomizer or DefaultRandomizer()
        self.available_types = MONSTER_TYPES.single_types()
        if restrict_types:
            self.restrict_monster_types()
        self.deck: 'CardList' = PrintingPress.make_cards(self.available_types, include_graveyard)
        self.hero_pool = [hero_type() for hero_type in VALHALLA if
                          hero_type.pool in self.available_types or hero_type.pool == MONSTER_TYPES.ALL]
        self.losers = []
        self.game_state = GameState.HERO_SELECTION

    def restrict_monster_types(self):
        for _ in range(2):
            monster_types = self.randomizer.select_monster_type(self.available_types, 0)
            self.available_types.remove(monster_types)

    def select_three_heroes(self):
        hero_choices = []
        for _ in range(3):
            hero = self.randomizer.select_hero(self.hero_pool)
            hero_choices.append(hero)
            self.hero_pool.remove(hero)
        return hero_choices

    def add_player(self, name: str) -> Player:
        assert self.game_state == GameState.HERO_SELECTION
        hero_choices = self.select_three_heroes()
        player = Player(self, name, hero_choices)
        self.players[name] = player
        return player

    def add_player_with_hero(self, name: str, hero: Hero = None) -> Player:
        assert self.game_state == GameState.HERO_SELECTION
        if hero is not None:
            self.hero_pool = [remaining_hero for remaining_hero in self.hero_pool if type(remaining_hero) != type(hero)]
        player = Player.new_player_with_hero(self, name, hero)
        self.players[name] = player
        return player

    def buying_step(self):
        assert len(self.players) % 2 == 0, "Must have an even number of players"
        assert self.game_state in (GameState.HERO_SELECTION, GameState.COMBAT_PHASE)
        self.game_state = GameState.BUY_PHASE
        self._generate_pairings()
        for player_name, player in self.players.items():
            if player.dead:
                continue
            player.buying_step()

    def combat_step(self):
        assert self.game_state == GameState.BUY_PHASE
        self.game_state = GameState.COMBAT_PHASE
        for player_name, player in self.players.items():
            player.at_buy_end()
            player.broadcast_buy_phase_event(events.BuyEndEvent())
        for player_1, player_2 in self.current_player_pairings:
            combat.fight_boards(WarParty(player_1), WarParty(player_2), self.randomizer)
        self.resolve_player_deaths()
        self._update_losers()
        self.turn_count += 1

    def get_paired_opponent(self, player: Player) -> Player:
        for p1, p2 in self.current_player_pairings:
            if p1 == player:
                return p2
            if p2 == player:
                return p1
        raise IndexError("Player not found in tavern pairings {}".format(player))

    def _generate_pairings(
            self):  # TODO figure out algorithm for ded guy someone in the bottom 3 fights the last ded guy
        fighting_players = [player for player in self.players.values() if not player.dead]
        if len(fighting_players) % 2 != 0:
            last_dead_player = self.losers[-1][1]
            fighting_players.append(last_dead_player)

        self.current_player_pairings = self.randomizer.select_player_pairings(fighting_players)
        logging.debug(f"Paired players: {self.current_player_pairings}")

    def _update_losers(self):
        for name, player in self.players.items():
            if (name, player) not in self.losers and player.health <= 0:
                self.losers.append((name, player))
        if len(self.losers) == len(self.players) - 1:
            for name, player in self.players.items():
                if (name, player) not in self.losers:
                    self.losers.append((name, player))
        if self.turn_count > self._max_turn_count:
            remaining_players = [(name, player) for name, player in self.players.items() if
                                 (name, player) not in self.losers]
            self.losers.extend(sorted(remaining_players, key=lambda e: e[1].health))

    def game_over(self):
        return len(self.losers) >= len(self.players)

    def resolve_player_deaths(self):
        for player in self.players.values():
            if player.health <= 0 and not player.dead:
                player.resolve_death()


class GameState(enum.Enum):
    HERO_SELECTION = 1
    BUY_PHASE = 2
    COMBAT_PHASE = 3
