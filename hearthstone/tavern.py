from typing import Dict

from hearthstone import combat, events
from hearthstone.cards import CardList, CardEvent, PrintingPress
from hearthstone.combat import WarParty
from hearthstone.hero import Hero
from hearthstone.player import Player
from hearthstone.randomizer import DefaultRandomizer


class Tavern:
    def __init__(self):
        self.players: Dict[str, Player] = {}
        self.deck: CardList = PrintingPress.make_cards()
        self.turn_count = 0
        self.current_player_pairings = []
        self.randomizer = DefaultRandomizer()
        self.losers = []

    def add_player(self, name: str, hero: Hero = None) -> Player:
        player = Player(self, name, hero)
        self.players[name] = player
        return player

    def buying_step(self):
        self.generate_pairings()
        for player_name, player in self.players.items():
            player.apply_turn_start_income()
            player.draw()
            player.hero.on_buy_step()
            player.broadcast_buy_phase_event(CardEvent(None, events.BUY_START))

    def combat_step(self):
        for player_name, player in self.players.items():
            player.broadcast_buy_phase_event(CardEvent(None, events.BUY_END))
        for player_1, player_2 in self.current_player_pairings:
            combat.fight_boards(WarParty(player_1), WarParty(player_2), self.randomizer)
        self.turn_count += 1

    def generate_pairings(self):
        fighting_players = [player for player in self.players.values() if player.health > 0]
        if len(fighting_players) % 2 != 0:
            last_dead_player = self.losers[-1][1]
            fighting_players.append(last_dead_player)

        self.current_player_pairings = self.randomizer.select_player_pairings(fighting_players)

    def update_losers(self):
        for name, player in self.players.items():
            if (name, player) not in self.losers and player.health <= 0:
                self.losers.append((name, player))
        if len(self.losers) == 7:
            for name, player in self.players.items():
                if (name, player) not in self.losers:
                    self.losers.append((name, player))

    def game_over(self):
        self.update_losers()
        return len(self.losers) >= len(self.players)
