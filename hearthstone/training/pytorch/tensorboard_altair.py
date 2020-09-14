from typing import List

import altair as alt
import pandas as pd
import tensorboard_vega_embed.summary
import torch
from torch.utils.tensorboard import SummaryWriter

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
        self.actions = []
        self.action_types = []
        self.rewards = []
        self.turn_counts = []
        self.boards = []
        self.hands = []
        self.stores = []

    def on_buy_phase_action(self, player: 'Player', action: Action, policy: torch.Tensor, value: torch.Tensor):
        self.update_gamestate(player, value, None)
        self.actions.append(action.str_in_context(player))
        self.action_types.append(type(action).__name__)

    def update_gamestate(self, player: 'Player', value, reward):
        self.turn_counts.append(player.tavern.turn_count)
        self.healths.append(player.health)
        self.avg_enemy_healths.append((sum(max(p.health, 0) for name, p in player.tavern.players.items()) - player.health) / 7.0)
        self.dead_players.append(len(player.tavern.losers)-3.5)
        if value is None:
            self.values.append(None)
        else:
            self.values.append(float(value))
        self.rewards.append(reward)
        self.boards.append([str(card) for card in player.in_play])
        self.hands.append([str(card) for card in player.hand])
        self.stores.append([str(card) for card in player.store])

    @staticmethod
    def _card_list_chart(name: str, cards_list: List[List[str]], selection):
        df = pd.DataFrame({
            name: cards_list,
        })
        df = df.reset_index().rename(columns={'index': 'step_in_game'}).explode(name).dropna()

        ranked_text = alt.Chart(df).mark_text().encode(
            y=alt.Y('row_number:O', axis=None)
        ).transform_lookup(lookup="step_in_game",
                           from_=alt.LookupSelection(key="step_in_game",
                                                     selection="gamestep_hover",
                                                     fields=["step_in_game"]),
                           as_="looked_up_step"
                           ).transform_filter(
            (alt.datum.step_in_game == alt.datum.looked_up_step)
        ).transform_window(
            row_number='row_number()'
        )
        return ranked_text.encode(text=f'{name}:N')

    def on_game_over(self, player: 'Player', ranking: int):
        self.update_gamestate(player, None, 3.5 - ranking)
        self.actions.append(None)
        self.action_types.append(None)
        df = pd.DataFrame({
            "turn_count": self.turn_counts,
            "health": self.healths,
            "dead_players": self.dead_players,
            "avg_enemy_health": self.avg_enemy_healths,
            "critic_value": self.values,
            "reward": self.rewards,
            "action": self.actions,
            "action_type": self.action_types,
        })

        selection = alt.selection_single(name="gamestep_hover", fields=['step_in_game'], empty="none")

        df = df.reset_index().rename(columns={'index': 'step_in_game'})
        melted = df.melt(id_vars=['step_in_game', 'action', 'action_type', 'turn_count'], value_name='value')

        base = alt.Chart(melted).encode(
            alt.X('step_in_game:Q', axis=alt.Axis()),
            tooltip=['step_in_game', 'action', 'variable', 'value'],
            color=alt.Color('variable'),
        ).properties(width=1000)

        val_chart = base.transform_filter(
            (alt.datum.variable != "reward")
        ).mark_line(point=True, size=3).encode(
            alt.Y('value'),
        ).add_selection(selection)

        reward_chart = base.transform_filter(
            (alt.datum.variable == "reward")
        ).mark_point(size=100, shape="diamond", filled=True).encode(
            alt.Y('value'),
            tooltip=['step_in_game', 'variable', 'value'],
        )

        turn_chart = base.transform_filter(
            (alt.datum.action_type == "EndPhaseAction")
        ).mark_rule().encode(
            x='step_in_game'
        )

        game_progression_chart = alt.layer(val_chart, reward_chart, turn_chart)

        action_chart = alt.Chart(df).mark_text(align='left', angle=270).encode(
            alt.X('step_in_game:Q'),
            y=alt.value(0),
            text='action',
            color='action_type'
        ).properties(width=999)

        board_chart = self._card_list_chart('board', self.boards, selection).properties(title='On Board', width=200)
        hand_chart = self._card_list_chart('hand', self.hands, selection).properties(title='In Hand', width=200)
        store_chart = self._card_list_chart('store', self.hands, selection).properties(title='In Store', width=200)

        left_chart = alt.vconcat(game_progression_chart, action_chart).resolve_legend('independent')
        full_chart = alt.hconcat(left_chart, board_chart, hand_chart, store_chart)
        json = full_chart.to_json()
        with open('/tmp/foo.txt', 'w') as f:
            f.write(json)
        tensorboard_vega_embed.summary.vega_embed(self.tensorboard, "GameSummary", json, self.global_step_context.get_global_step())
