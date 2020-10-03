import collections
from collections import defaultdict
from typing import List, Optional, Dict

import altair as alt
import pandas as pd
import tensorboard_vega_embed.summary
import torch
from torch.utils.tensorboard import SummaryWriter

from hearthstone.simulator.agent import Action, SellAction, SummonAction, BuyAction, generate_valid_actions
from hearthstone.training.pytorch import hearthstone_state_encoder
from hearthstone.training.pytorch.hearthstone_state_encoder import encode_player, encode_valid_actions, get_action_index
from hearthstone.training.pytorch.replay_buffer import ReplayBuffer
from hearthstone.training.pytorch.surveillance import Parasite, GlobalStepContext


class GAEPlotter(Parasite):
    GameStep = collections.namedtuple("GameStep", ["state", "action", "valid_actions", "action_prob", "value"])

    def __init__(self, lam: float = 0.9, gamma: float = 0.99,
                 device: Optional[torch.device] = None):
        """
        Puts transitions into the replay buffer.

        Args:
            replay_buffer: Buffer of transitions.
        """
        self.lam = lam
        self.gamma = gamma
        self.device = device

        self.game_steps = []

    def on_buy_phase_action(self, player: 'Player', action: Action, policy: torch.Tensor, value: torch.Tensor):
        action_index = get_action_index(action)
        self.game_steps.append(
            self.GameStep(
                state=encode_player(player, self.device),
                action=int(action_index),
                valid_actions=encode_valid_actions(player, self.device),
                action_prob=float(policy[0][action_index]),
                value=float(value)
            )
        )

    def on_game_over(self, player: 'Player', ranking: int) -> Dict[str, List]:
        gae_returns = []
        returns = []
        retn = 3.5 - ranking
        gae_return = retn
        next_value = retn
        for i in range(len(self.game_steps) - 1, -1, -1):
            game_step = self.game_steps[i]
            gae_returns.append(gae_return)
            returns.append(retn)
            gae_return = next_value + (gae_return - next_value) * self.gamma * self.lam
            next_value = self.gamma * game_step.value
            retn *= self.gamma

        gae_returns.reverse()
        gae_returns.append(None)
        returns.reverse()
        returns.append(None)
        return {
            "critic_value_gae_return": gae_returns,
            "return": returns,
        }


class TensorboardAltairPlotter(Parasite):
    def __init__(self, tensorboard: SummaryWriter, global_step_context: GlobalStepContext):
        if not global_step_context.should_plot():
            self.dont_plot = True
            return
        self.dont_plot = False
        self.tensorboard = tensorboard
        self.healths = []
        self.coins = []
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
        self.basic_action_probs = []
        self.sell_probs = []
        self.summon_probs = []
        self.buy_probs = []

        self.gae_plotter = GAEPlotter()

    def populate_action_probs(self, player: 'player', policy:torch.Tensor):
        policy = policy.detach().squeeze().exp()
        self.basic_action_probs.append([float(policy[hearthstone_state_encoder.get_action_index(action)]) for action in
                                        hearthstone_state_encoder.ALL_ACTIONS.player_action_set])

        action_probs = defaultdict(lambda: 0.0)
        for action in generate_valid_actions(player):
            if str(action) in hearthstone_state_encoder.ALL_ACTIONS_DICT:
                if isinstance(action, SellAction):
                    card = player.in_play[action.index]
                elif isinstance(action, SummonAction):
                    card = player.hand[action.index]
                elif isinstance(action,BuyAction):
                    card = player.store[action.index]
                else:
                    continue

                action_probs[card] += float(policy[hearthstone_state_encoder.get_action_index(action)])

        self.sell_probs.append([action_probs[card] for card in player.in_play])
        self.summon_probs.append([action_probs[card] for card in player.hand])
        self.buy_probs.append([action_probs[card] for card in player.store])

    def on_buy_phase_action(self, player: 'Player', action: Action, policy: torch.Tensor, value: torch.Tensor):
        if self.dont_plot:
            return
        self.update_gamestate(player, value, None)
        self.actions.append(action.str_in_context(player))
        self.action_types.append(type(action).__name__)
        self.populate_action_probs(player, policy)
        self.gae_plotter.on_buy_phase_action(player, action, policy, value)

    def update_gamestate(self, player: 'Player', value, reward):
        self.turn_counts.append(player.tavern.turn_count)
        self.healths.append(player.health)
        self.coins.append(player.coins)
        self.avg_enemy_healths.append(
            (sum(max(p.health, 0) for name, p in player.tavern.players.items()) - player.health) / 7.0)
        self.dead_players.append(len(player.tavern.losers) - 3.5)
        if value is None:
            self.values.append(None)
        else:
            self.values.append(float(value))
        self.rewards.append(reward)
        self.boards.append([str(card) for card in player.in_play])
        self.hands.append([str(card) for card in player.hand])
        self.stores.append([str(card) for card in player.store])

    @staticmethod
    def _action_chart(df: pd.DataFrame, name: str, max_size: int):
        ranked_text = alt.Chart(df).mark_text().encode(
            y=alt.Y('row_number:O', axis=None, scale=alt.Scale(domain=list(range(1, max_size+1)))),
            color=alt.Color("action_probability:Q", scale=alt.Scale(domain=[0, 1], scheme="bluegreen")),
            tooltip=["action_probability:Q"]
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

    @staticmethod
    def _card_list_chart(name: str, cards_list: List[List[str]], action_probs: List[List[str]], max_size:int):
        df = pd.DataFrame({
            name: cards_list,
            "action_probability": action_probs
        })
        df = df.apply(pd.Series.explode).reset_index().rename(columns={'index': 'step_in_game'}).dropna()
        return TensorboardAltairPlotter._action_chart(df, name, max_size)

    @staticmethod
    def _player_action_chart(action_probs: List[List[str]], max_size: int):
        basic_actions = [str(action) for action in hearthstone_state_encoder.ALL_ACTIONS.player_action_set]
        df = pd.DataFrame(
            {
                "basic_actions": [basic_actions] * len(action_probs),
                "action_probability": action_probs,
            }
        )
        df = df.apply(pd.Series.explode).reset_index().rename(columns={'index': 'step_in_game'})
        return TensorboardAltairPlotter._action_chart(df, "basic_actions", max_size)

    def on_game_over(self, player: 'Player', ranking: int):
        if self.dont_plot:
            return
        self.update_gamestate(player, None, 3.5 - ranking)
        self.actions.append(None)
        self.action_types.append(None)
        self.sell_probs.append([None for _ in player.in_play])
        self.summon_probs.append([None for _ in player.hand])
        self.buy_probs.append([None for _ in player.store])

        columns = self.gae_plotter.on_game_over(player, ranking)
        columns.update({
            "turn_count": self.turn_counts,
            "health": self.healths,
            "coins": self.coins,
            "dead_players": self.dead_players,
            "avg_enemy_health": self.avg_enemy_healths,
            "critic_value": self.values,
            "reward": self.rewards,
            "action": self.actions,
            "action_type": self.action_types,
        })
        df = pd.DataFrame(columns)
        hover_selection = alt.selection_single(name="gamestep_hover", fields=['step_in_game'], encodings=['x'], empty="none",
                                         on="mousemove", nearest=True)
        legend_selection = alt.selection_multi(name="scalar_legend", fields=['variable'], bind="legend")

        df = df.reset_index().rename(columns={'index': 'step_in_game'})
        melted = df.melt(id_vars=['step_in_game', 'action', 'action_type', 'turn_count'], value_name='value')
        base = alt.Chart(melted)

        rule = base.transform_filter(hover_selection).mark_rule().encode(alt.X('step_in_game:Q'))
        value_base = base.encode(
            alt.X('step_in_game:Q'),
            color=alt.Color('variable:N'),
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.2)),
        ).properties(width=1000)

        point_chart = value_base.mark_point(

        ).encode(
            alt.Y('value:Q'),
            opacity=alt.condition(hover_selection, alt.value(1), alt.value(0)),
            tooltip=['step_in_game:Q', 'action:N', 'variable:N', 'value:Q']
        ).add_selection(
            hover_selection).add_selection(legend_selection)
        text_chart = value_base.mark_text(align='left', dx=5, dy=-7
                                          ).encode(alt.Y('value:Q'),
                                                   text=alt.condition(hover_selection,
                                                                      alt.Text('value:Q',
                                                                               format='.3'),
                                                                      alt.value('')))
        line_chart = value_base.transform_filter(
            (alt.datum.variable != "reward")
        ).mark_line().encode(
            alt.Y('value:Q'),
        )
        reward_chart = value_base.transform_filter(
            (alt.datum.variable == "reward")
        ).mark_point(size=100, shape="diamond", filled=True).encode(
            alt.Y('value:Q'),
            tooltip=['step_in_game:Q', 'variable:N', 'value:Q'],
        )

        turn_chart = value_base.transform_filter(
            (alt.datum.action_type == "EndPhaseAction")
        ).mark_rule().encode(
            alt.X('step_in_game:Q')
        )

        game_progression_chart = alt.layer(rule, point_chart, text_chart, line_chart, reward_chart, turn_chart)

        action_chart = alt.Chart(df).mark_text(align='left', angle=270
                                               ).transform_lookup(lookup="step_in_game",
                                                                  from_=alt.LookupSelection(
                                                                      key="step_in_game",
                                                                      selection="gamestep_hover",
                                                                      fields=["step_in_game"]),
                                                                  as_="looked_up_step"
                                                                  ).encode(
            alt.X('step_in_game:Q'),
            y=alt.value(0),
            text='action:N',
            color='action_type:N',
            opacity=alt.condition((alt.datum.step_in_game == alt.datum.looked_up_step), alt.value(1), alt.value(0.4))
        ).properties(width=999)

        basic_action_chart = self._player_action_chart(self.basic_action_probs, len(hearthstone_state_encoder.ALL_ACTIONS.player_action_set)).properties(title='Basic Actions', width=400)
        board_chart = self._card_list_chart('board', self.boards, self.sell_probs, 7).properties(title='On Board', width=400)
        hand_chart = self._card_list_chart('hand', self.hands, self.summon_probs, 10).properties(title='In Hand', width=400)
        store_chart = self._card_list_chart('store', self.stores, self.buy_probs, 7).properties(title='In Store', width=400)

        left_chart = alt.vconcat(game_progression_chart, action_chart).resolve_scale(color='independent')
        right_chart = alt.vconcat(basic_action_chart, board_chart, hand_chart, store_chart).resolve_legend('shared')
        full_chart = alt.hconcat(left_chart, right_chart)
        json = full_chart.to_json()

        tensorboard_vega_embed.summary.vega_embed(self.tensorboard, "GameSummary", json,
                                                  self.global_step_context.get_global_step())
