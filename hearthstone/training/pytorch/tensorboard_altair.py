import itertools
from collections import defaultdict
from typing import List, NamedTuple, Any, Optional

import altair as alt
import pandas as pd
import tensorboard_vega_embed.summary
import torch
from torch.utils.tensorboard import SummaryWriter

from hearthstone.simulator.agent.actions import StandardAction, SellAction, BuyAction, Action
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.simulator.replay.observer import Observer, Annotation
from hearthstone.simulator.replay.replay import Replay, ReplayStep
from hearthstone.training.pytorch.encoding import default_encoder
from hearthstone.training.pytorch.encoding.default_encoder import EncodedActionSet, DefaultEncoder
from hearthstone.training.common.state_encoding import SummonComponent
from hearthstone.training.pytorch.surveillance import GlobalStepContext


class TensorboardAltairAnnotation(NamedTuple):
    """
    Additional information that the TensorboardAltair plotter needs to generate nice game graphs.  This information
    will be attached to each game step.
    """
    turn_count: int
    health: int
    coins: int
    avg_enemy_health: float
    dead_players: float
    board: List[str]
    hand: List[str]
    store: List[str]
    action: str
    action_type: str
    hero: str


class TensorboardAltairAnnotator(Observer):
    """
    This observer annotates the replay with extra information for the purposes of generating nice Tensorboard plots.
    """

    def __init__(self, player_names: List[str]):
        """

        :param player_names: The players to annotate for.  All other player actions will be ignored.
        """
        self.player_names = set(player_names)

    def name(self) -> str:
        return "TensorboardAltairAnnotator"

    def on_action(self, tavern: 'Tavern', player_name: str, action: 'Action') -> Annotation:

        if player_name not in self.player_names:
            return None

        if not isinstance(action, StandardAction):
            return None
        player = tavern.players[player_name]
        return TensorboardAltairAnnotation(
            turn_count=player.tavern.turn_count,
            health=player.health,
            coins=player.coins,
            avg_enemy_health=(sum(max(p.health, 0) for name, p in tavern.players.items()) - max(player.health,
                                                                                                0)) / (
                                     len(tavern.players) - 1),
            dead_players=len(tavern.losers) - (len(tavern.players) - 1) / 2.0,
            board=[str(card) for card in player.in_play],
            hand=[str(card) for card in player.hand],
            store=[str(card) for card in player.store],
            action=action.str_in_context(player),
            action_type=type(action).__name__,
            hero=str(player.hero)
        )


def _action_chart(df: pd.DataFrame, name: str, action_name: str, max_size: int):
    ranked_text = alt.Chart(df).mark_text().transform_lookup(lookup="step_in_game",
                                                             from_=alt.LookupSelection(key="step_in_game",
                                                                                       selection="gamestep_hover",
                                                                                       fields=["step_in_game"]),
                                                             as_="looked_up_step"
                                                             ).transform_filter(
        (alt.datum.step_in_game == alt.datum.looked_up_step)
    ).transform_window(
        row_number='row_number()'
    )
    tooltip_fields = [alt.Tooltip("action_probability:Q", title="Probability of {}".format(action_name))]
    if 'rearrange_logits' in df:
        tooltip_fields.append(alt.Tooltip("rearrange_logits:Q", title="Positioning"))

    return ranked_text.encode(text=f'{name}:N',
                              y=alt.Y('row_number:O', axis=None,
                                      scale=alt.Scale(domain=list(range(1, max_size + 1)))),
                              color=alt.Color("action_probability:Q",
                                              scale=alt.Scale(domain=[0, 1], scheme="goldred")),
                              tooltip=tooltip_fields)


def _card_list_chart(name: str, action_name: str, cards_list: List[List[str]], action_probs: List[List[float]],
                     max_size: int,
                     rearrange_logits: Optional[List[List[float]]] = None):
    df = pd.DataFrame({
        name: cards_list,
        "action_probability": action_probs,
    })

    if rearrange_logits:
        df['rearrange_logits'] = rearrange_logits
    df = df.apply(pd.Series.explode).reset_index().rename(columns={'index': 'step_in_game'}).dropna(subset=[name])
    return _action_chart(df, name, action_name, max_size)


def _player_action_chart(action_probs: List[List[float]], max_size: int):
    basic_actions = [str(action) for action in default_encoder.ALL_ACTIONS.player_action_set]
    df = pd.DataFrame(
        {
            "basic_actions": [basic_actions] * len(action_probs),
            "action_probability": action_probs,
        }
    )
    df = df.apply(pd.Series.explode).reset_index().rename(columns={'index': 'step_in_game'})
    return _action_chart(df, "basic_actions", "action", max_size)


def calc_action_probs(policy: torch.Tensor, valid_actions: EncodedActionSet, store: List[str], hand: List[str],
                      board: List[str]) -> (List, List, List, List):
    with torch.no_grad():
        encoder = DefaultEncoder()
        flat_valid_actions = torch.cat(
            (valid_actions.player_action_tensor.flatten(0), valid_actions.card_action_tensor.flatten(0), valid_actions.spell_action_tensor.flatten(0)), dim=0)
        policy = policy.to("cpu").masked_fill(flat_valid_actions.to("cpu").logical_not(), -1e30).squeeze().exp()
        basic_action_probs = [float(policy[encoder.get_action_component_index(action)]) for action in
                              default_encoder.ALL_ACTIONS.player_action_set]

    action_probs = defaultdict(lambda: 0.0)
    for action_component in itertools.chain(default_encoder.ALL_ACTIONS.player_action_set,
                                            *default_encoder.ALL_ACTIONS.card_action_set):
        if isinstance(action_component, SellAction):
            card = f"B{action_component.index}"
        elif isinstance(action_component, SummonComponent):
            card = f"H{action_component.index}"
        elif isinstance(action_component, BuyAction):
            card = f"S{action_component.index}"
        else:
            continue
        action_probs[card] += float(policy[encoder.get_action_component_index(action_component)])
    buy_probs = [action_probs[f"S{index}"] for index in range(len(store))]
    summon_probs = [action_probs[f"H{index}"] for index in range(len(hand))]
    sell_probs = [action_probs[f"B{index}"] for index in range(len(board))]
    return basic_action_probs, buy_probs, summon_probs, sell_probs


def plot_replay(replay: Replay, player_name: str, tensorboard: SummaryWriter, global_step_context: GlobalStepContext):
    if not global_step_context.should_plot():
        return
    game_steps: List[ReplayStep] = [step for step in replay.steps if step.player == player_name]

    def valid_step(step: ReplayStep) -> Any:
        return step.player == player_name and isinstance(step.action, StandardAction)

    scalar_columns = ["turn_count", "health", "coins", "avg_enemy_health", "dead_players", "action", "action_type"]
    columns = {
        field: [getattr(step.observer_annotations.get("TensorboardAltairAnnotator"), field) for step in game_steps if
                valid_step(step)]
        for field in
        scalar_columns
    }

    columns.update({
        "critic_value": [step.agent_annotation.value for step in game_steps if
                         valid_step(step)],
        "reward": [step.agent_annotation.gae_info.reward if step.agent_annotation.gae_info.is_terminal else None for
                   step in game_steps if
                   valid_step(step)],
        "critic_value_gae_return": [step.agent_annotation.gae_info.gae_return for step in game_steps if
                                    valid_step(step)],
        "advantage": [step.agent_annotation.gae_info.gae_return - step.agent_annotation.value for step in game_steps if
                      valid_step(step)],
        "return": [step.agent_annotation.gae_info.retrn for step in game_steps if
                   valid_step(step)],
    })

    stores = []
    hands = []
    boards = []
    basic_action_probs = []
    buy_probs = []
    summon_probs = []
    sell_probs = []
    rearrange_logits = []
    for step in replay.steps:
        if not valid_step(step):
            continue

        annotations: TensorboardAltairAnnotation = step.observer_annotations.get("TensorboardAltairAnnotator")
        basic_action_prob, buy_prob, summon_prob, sell_prob = calc_action_probs(
            step.agent_annotation.debug.component_policy,
            step.agent_annotation.valid_actions,
            annotations.store,
            annotations.hand,
            annotations.board,
            )
        rearrange_logit = step.agent_annotation.debug.permutation_logits.squeeze(0).tolist()
        assert len(rearrange_logit) >= len(annotations.board)
        rearrange_logit = rearrange_logit[:len(annotations.board)]
        stores.append(annotations.store)
        hands.append(annotations.hand)
        boards.append(annotations.board)
        basic_action_probs.append(basic_action_prob)
        buy_probs.append(buy_prob)
        summon_probs.append(summon_prob)
        sell_probs.append(sell_prob)
        rearrange_logits.append(rearrange_logit)

    df = pd.DataFrame(columns)
    hover_selection = alt.selection_single(name="gamestep_hover", fields=['step_in_game'], encodings=['x'],
                                           empty="none",
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

    basic_action_chart = _player_action_chart(basic_action_probs, len(
        default_encoder.ALL_ACTIONS.player_action_set)).properties(title='Basic Actions: ' + annotations.hero,
                                                                   width=400)
    board_chart = _card_list_chart('board', 'sell', boards, sell_probs, 7, rearrange_logits).properties(
        title='On Board', width=400)
    hand_chart = _card_list_chart('hand', 'summon', hands, summon_probs, 10).properties(title='In Hand', width=400)
    store_chart = _card_list_chart('store', 'buy', stores, buy_probs, 7).properties(title='In Store', width=400)

    left_chart = alt.vconcat(game_progression_chart, action_chart).resolve_scale(color='independent')
    right_chart = alt.vconcat(basic_action_chart, board_chart, hand_chart, store_chart).resolve_legend('shared')
    full_chart = alt.hconcat(left_chart, right_chart).configure(autosize=alt.AutoSizeParams(resize=True))
    json = full_chart.to_json()
    tensorboard_vega_embed.summary.vega_embed(tensorboard, "GameSummary", json,
                                              global_step_context.get_global_step())
