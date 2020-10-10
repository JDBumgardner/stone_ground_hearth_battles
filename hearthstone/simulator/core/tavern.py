import enum
from typing import Dict

from hearthstone.simulator.core import hero, combat, events
from hearthstone.simulator.core.cards import CardList, PrintingPress
from hearthstone.simulator.core.combat import WarParty
from hearthstone.simulator.core.hero import Hero
from hearthstone.simulator.core.player import Player
from hearthstone.simulator.core.randomizer import DefaultRandomizer


class Tavern:
    def __init__(self):
        self.players: Dict[str, Player] = {}
        self.deck: CardList = PrintingPress.make_cards()
        self.hero_pool = [hero_type() for hero_type in hero.VALHALLA]
        self.turn_count = 0
        self._max_turn_count = 50
        self.current_player_pairings = []
        self.randomizer = DefaultRandomizer()
        self.losers = []
        self.game_state = GameState.HERO_SELECTION

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

    def add_player_with_hero(self, name: str, hero: Hero=None) -> Player:
        assert self.game_state == GameState.HERO_SELECTION
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
            player.decrease_tavern_upgrade_cost()
            player.broadcast_buy_phase_event(events.BuyEndEvent())
        for player_1, player_2 in self.current_player_pairings:
            combat.fight_boards(WarParty(player_1), WarParty(player_2), self.randomizer)
        self.resolve_player_deaths()
        self._update_losers()
        self.turn_count += 1

    def _generate_pairings(self): #TODO figure out algorithm for ded guy someone in the bottom 3 fights the last ded guy
        fighting_players = [player for player in self.players.values() if player.health > 0]
        if len(fighting_players) % 2 != 0:
            last_dead_player = self.losers[-1][1]
            fighting_players.append(last_dead_player)

        self.current_player_pairings = self.randomizer.select_player_pairings(fighting_players)

    def _update_losers(self):
        for name, player in self.players.items():
            if (name, player) not in self.losers and player.health <= 0:
                self.losers.append((name, player))
        if len(self.losers) == len(self.players) - 1:
            for name, player in self.players.items():
                if (name, player) not in self.losers:
                    self.losers.append((name, player))
        if self.turn_count > self._max_turn_count:
            remaining_players = [(name, player) for name, player in self.players.items() if (name, player) not in self.losers]
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