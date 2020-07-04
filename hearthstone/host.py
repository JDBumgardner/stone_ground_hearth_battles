from typing import Dict

from hearthstone.agent import Agent, EndPhaseAction
from hearthstone.tavern import Tavern


class RoundRobinHost:
    tavern: Tavern
    agents: Dict[str, Agent]

    def __init__(self, agents: Dict[str, Agent]):
        self.tavern = Tavern()
        self.agents = agents
        for player_name in agents.keys():
            self.tavern.add_player(player_name)

    def play_game(self):
        for player_name, player in self.tavern.players.items():
            player.choose_hero(self.agents[player_name].hero_choice_action(player))
        while not self.tavern.game_over():
            self.tavern.buying_step()
            for player_name, player in self.tavern.players.items():
                if player.health <= 0:
                    continue
                agent = self.agents[player_name]
                for _ in range(20):
                    action = agent.buy_phase_action(player)
                    action.apply(player)
                    if player.discovered_cards:
                        discovered_card = agent.discover_choice_action(player)
                        player.select_discover(discovered_card)

                    if type(action) is EndPhaseAction:
                        break
                if len(player.in_play) > 1:
                    arrangement = agent.rearrange_cards(player)
                    assert set(arrangement) == set(player.in_play)
                    player.in_play = arrangement
            self.tavern.combat_step()
