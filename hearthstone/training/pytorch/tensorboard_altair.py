import collections

import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import altair as alt

import tensorboard_vega_embed.summary

from hearthstone.agent import Action
from hearthstone.training.pytorch.surveillance import Parasite, GlobalStepContext


class TensorboardAltairPlotter(Parasite):
    def __init__(self, tensorboard: SummaryWriter, global_step_context: GlobalStepContext):
        self.tensorboard = tensorboard
        self.healths = []
        self.dead_players = []
        self.avg_enemy_healths = []
        self.values = []
        self.global_step_context = global_step_context
        self.action_types = []
        self.rewards = []

    def on_buy_phase_action(self, player: 'Player', action: Action, policy: torch.Tensor, value: torch.Tensor):
        self.update_gamestate(player, value, None)
        self.action_types.append(type(action).__name__)

    def update_gamestate(self, player: 'Player', value, reward):
        self.healths.append(player.health)
        self.avg_enemy_healths.append((sum(max(p.health, 0) for name, p in player.tavern.players.items()) - player.health) / 7.0)
        self.dead_players.append(len(player.tavern.losers)-3.5)
        if value is None:
            self.values.append(None)
        else:
            self.values.append(float(value))
        self.rewards.append(reward)

    def on_game_over(self, player: 'Player', ranking: int):
        self.update_gamestate(player, None, 3.5 - ranking)
        df = pd.DataFrame({"health": self.healths,
                           "dead_players": self.dead_players,
                           "avg_enemy_health":self.avg_enemy_healths,
                           "critic_value": self.values,
                           "reward": self.rewards})
        df = df.reset_index().melt(id_vars='index', value_name='value')
        base = alt.Chart(df).encode(
            alt.X('index:Q'),
            tooltip=['index', 'variable', 'value'],
            color=alt.Color('variable'),
        )

        val_chart = base.transform_filter(
            (alt.datum.variable != "reward")
        ).mark_line(point=True, size=3).encode(
            alt.Y('value'),
        )

        reward_chart = base.transform_filter(
            (alt.datum.variable == "reward")
        ).mark_point(size=100, shape="diamond", filled=True).encode(
            alt.Y('value'),
            tooltip=['index', 'variable', 'value'],
        )

        json = alt.layer(val_chart, reward_chart).interactive().to_json()
        tensorboard_vega_embed.summary.vega_embed(self.tensorboard, "GameSummary", json, self.global_step_context.get_global_step())
