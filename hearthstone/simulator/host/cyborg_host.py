import asyncio

from hearthstone.battlebots.early_game_bot import EarlyGameBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.simulator.agent import EndPhaseAction
from hearthstone.simulator.host import AsyncHost


class CyborgArena(AsyncHost):
    async def _async_play_round(self):
        self.tavern.buying_step()

        async def perform_player_actions(agent, player):
            for _ in range(20):
                if player.discover_queue:
                    try:
                        discovered_card = await agent.discover_choice_action(player)
                    except ConnectionError:
                        print("replace with a bot")
                        # replace the agent and player
                        agent = PriorityFunctions.battlerattler_priority_bot(3, EarlyGameBot)
                        self.agents[player.name] = agent
                        discovered_card = await agent.discover_choice_action(player)

                    player.select_discover(discovered_card)
                else:
                    try:
                        action = await agent.buy_phase_action(player)
                    except ConnectionError:
                        print("replace with a bot")

                        # replace the agent and player
                        agent = PriorityFunctions.battlerattler_priority_bot(3, EarlyGameBot)
                        self.agents[player.name] = agent
                        action = await agent.buy_phase_action(player)
                    action.apply(player)
                    if type(action) is EndPhaseAction:
                        break
            if len(player.in_play) > 1:
                try:
                    arrangement = await agent.rearrange_cards(player)
                except ConnectionError:
                    print("replace with a bot")
                    # replace the agent and player
                    agent = PriorityFunctions.battlerattler_priority_bot(3, EarlyGameBot)
                    self.agents[player.name] = agent
                    arrangement = await agent.rearrange_cards(player)
                player.rearrange_cards(arrangement)

        perform_player_action_tasks = []
        for player_name, player in self.tavern.players.items():
            if player.dead:
                continue
            perform_player_action_tasks.append(
                asyncio.create_task(perform_player_actions(self.agents[player_name], player)))
        await asyncio.gather(*perform_player_action_tasks)

        self.tavern.combat_step()
        if self.tavern.game_over():
            game_over_tasks = []
            for position, (name, player) in enumerate(reversed(self.tavern.losers)):
                game_over_tasks.append(asyncio.create_task(self.agents[name].game_over(player, position)))
            await asyncio.gather(*game_over_tasks)