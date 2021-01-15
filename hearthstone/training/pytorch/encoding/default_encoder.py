import enum
from typing import List, Optional, Dict

import numpy as np
import torch

from hearthstone.simulator.agent import TripleRewardsAction, TavernUpgradeAction, RerollAction, \
    EndPhaseAction, SummonAction, BuyAction, SellAction, StandardAction, DiscoverChoiceAction, HeroPowerAction
from hearthstone.simulator.core.cards import CardLocation, PrintingPress
from hearthstone.simulator.core.hero import VALHALLA
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.player import Player, StoreIndex, HandIndex, BoardIndex, DiscoverIndex
from hearthstone.training.pytorch.encoding.state_encoding import State, \
    LocatedCard, Feature, ScalarFeature, OnehotFeature, CombinedFeature, ListOfFeatures, SortedByValueFeature, \
    EncodedActionSet, Encoder, InvalidAction, ActionSet

MAX_ENCODED_STORE = 7
MAX_ENCODED_HAND = 10
MAX_ENCODED_BOARD = 7
MAX_ENCODED_DISCOVER = 3


def enum_to_int(value: Optional[enum.Enum]) -> int:
    if value is not None:
        return value.value
    else:
        return 0


CARD_TYPE_TO_INT = {card_type: ind for ind, card_type in enumerate(PrintingPress.cards)}
INT_TO_CARD_TYPE = {ind: card_type for ind, card_type in enumerate(PrintingPress.cards)}
HERO_TYPE_TO_INT = {hero_type: ind for ind, hero_type in enumerate(VALHALLA)}
INT_TO_HERO_TYPE = {ind: hero_type for ind, hero_type in enumerate(VALHALLA)}


def default_card_encoding() -> Feature:
    """
    Default encoder for type `LocatedCard`.
    """
    return CombinedFeature([
        ScalarFeature(lambda card: 1.0),  # Present
        ScalarFeature(lambda card: float(card.card.tier)),
        ScalarFeature(lambda card: float(card.card.attack)),
        ScalarFeature(lambda card: float(card.card.health)),
        ScalarFeature(lambda card: float(card.card.golden)),
        ScalarFeature(lambda card: float(card.card.taunt)),
        ScalarFeature(lambda card: float(card.card.divine_shield)),
        ScalarFeature(lambda card: float(card.card.poisonous)),
        ScalarFeature(lambda card: float(card.card.magnetic)),
        ScalarFeature(lambda card: float(card.card.windfury)),
        ScalarFeature(lambda card: float(card.card.reborn)),
        ScalarFeature(lambda card: float(bool(card.card.deathrattles))),
        ScalarFeature(lambda card: float(bool(card.card.battlecry))),
        OnehotFeature(lambda card: CARD_TYPE_TO_INT[type(card.card)], len(PrintingPress.cards) + 1),
        OnehotFeature(lambda card: enum_to_int(card.card.monster_type), len(MONSTER_TYPES) + 1),
        OnehotFeature(lambda card: enum_to_int(card.location), len(CardLocation) + 1),
    ])


def default_player_encoding() -> Feature:
    """
    Default encoder for the player level features (non-card features).

    Encodes a `Player`.
    """
    return CombinedFeature([
        ScalarFeature(lambda player: float(player.tavern.turn_count)),
        ScalarFeature(lambda player: float(player.health)),
        ScalarFeature(lambda player: float(player.coins)),
        ScalarFeature(lambda player: float(player.tavern_tier)),
        ScalarFeature(lambda player: float(len(player.in_play))),
        ScalarFeature(lambda player: float(len(player.hand))),
        ScalarFeature(lambda player: float(len(player.tavern))),
        OnehotFeature(lambda player: HERO_TYPE_TO_INT[type(player.hero)], len(VALHALLA) + 1),
        SortedByValueFeature(lambda player: [p.health for name, p in player.tavern.players.items()], 8),
    ])


def default_cards_encoding() -> Feature:
    """
    Default encoder for the card-level features.

    Encodes a `Player`.
    """
    return CombinedFeature([
        ListOfFeatures(
            lambda player: [LocatedCard(card, CardLocation.STORE) for card in player.store],
            default_card_encoding(), MAX_ENCODED_STORE),
        ListOfFeatures(
            lambda player: [LocatedCard(card, CardLocation.HAND) for card in player.hand],
            default_card_encoding(), MAX_ENCODED_HAND),
        ListOfFeatures(
            lambda player: [LocatedCard(card, CardLocation.BOARD) for card in player.in_play],
            default_card_encoding(), MAX_ENCODED_BOARD),
        ListOfFeatures(
            lambda player: [LocatedCard(card, CardLocation.DISCOVER) for card in
                            (player.discover_queue[0] if player.discover_queue else [])],
            default_card_encoding(), MAX_ENCODED_DISCOVER)
    ])


DEFAULT_PLAYER_ENCODING = default_player_encoding()
DEFAULT_CARDS_ENCODING = default_cards_encoding()


def store_indices() -> List[StoreIndex]:
    return [StoreIndex(i) for i in range(MAX_ENCODED_STORE)]


def hand_indices() -> List[HandIndex]:
    return [HandIndex(i) for i in range(MAX_ENCODED_HAND)]


def board_indices() -> List[BoardIndex]:
    return [BoardIndex(i) for i in range(MAX_ENCODED_BOARD)]


def discover_indices() -> List[DiscoverIndex]:
    return [DiscoverIndex(i) for i in range(MAX_ENCODED_DISCOVER)]


def _all_actions() -> ActionSet:
    player_action_set = [TripleRewardsAction(), TavernUpgradeAction(), RerollAction(), EndPhaseAction(False),
                         HeroPowerAction(), EndPhaseAction(True)]
    store_action_set = [
        [BuyAction(index), InvalidAction(), InvalidAction(),
         InvalidAction(), InvalidAction(), HeroPowerAction(store_target=index)] for index in store_indices()]
    hand_action_set = [
        [InvalidAction(), SummonAction(index), SummonAction(index, [BoardIndex(0)]),
         InvalidAction(), InvalidAction(), InvalidAction()] for index in hand_indices()]
    board_action_set = [
        [InvalidAction(), InvalidAction(), InvalidAction(),
         SellAction(index), InvalidAction(), HeroPowerAction(board_target=index)] for index in board_indices()]
    discover_action_set = [
        [InvalidAction(), InvalidAction(), InvalidAction(),
         InvalidAction(), DiscoverChoiceAction(index), InvalidAction()] for index in
        discover_indices()]
    return ActionSet(player_action_set, store_action_set + hand_action_set + board_action_set + discover_action_set)


ALL_ACTIONS = _all_actions()


def _all_actions_dict():
    result = {}
    index = 0
    for player_action in ALL_ACTIONS.player_action_set:
        result[str(player_action)] = index
        index += 1
    for card_actions in ALL_ACTIONS.card_action_set:
        for card_action in card_actions:
            result[str(card_action)] = index
            index += 1
    return result


ALL_ACTIONS_DICT: Dict[str, int] = _all_actions_dict()


class DefaultEncoder(Encoder):
    def encode_state(self, player: Player) -> State:
        player_tensor = torch.from_numpy(DEFAULT_PLAYER_ENCODING.encode(player))
        cards_tensor = torch.from_numpy(DEFAULT_CARDS_ENCODING.encode(player))
        return State(player_tensor, cards_tensor)

    def encode_valid_actions(self, player: Player, rearrange_phase: bool = False) -> EncodedActionSet:
        actions = ALL_ACTIONS

        player_action_tensor = torch.tensor([action.valid(player) for action in actions.player_action_set])
        cards_action_array = np.ndarray((len(actions.card_action_set), len(actions.card_action_set[0])), dtype=bool)
        for i, card_actions in enumerate(actions.card_action_set):
            for j, action in enumerate(card_actions):
                cards_action_array[i, j] = action.valid(player)
        cards_action_tensor = torch.from_numpy(cards_action_array)

        cards_to_rearrange = torch.tensor(
            [MAX_ENCODED_STORE + MAX_ENCODED_HAND, len(player.in_play)], dtype=torch.long)

        return EncodedActionSet(player_action_tensor, cards_action_tensor, torch.tensor(rearrange_phase),
                                cards_to_rearrange)

    def player_encoding(self) -> Feature:
        return DEFAULT_PLAYER_ENCODING

    def cards_encoding(self) -> Feature:
        return DEFAULT_CARDS_ENCODING

    def action_encoding_size(self) -> int:
        player_action_size = len(ALL_ACTIONS.player_action_set)
        card_action_size = len(ALL_ACTIONS.card_action_set) * len(ALL_ACTIONS.card_action_set[0])
        return player_action_size + card_action_size

    def get_action_index(self, action: StandardAction) -> int:
        return ALL_ACTIONS_DICT[str(action)]

    def get_indexed_action(self, index: int) -> StandardAction:
        if index < len(ALL_ACTIONS.player_action_set):
            return ALL_ACTIONS.player_action_set[index]
        else:
            card_action_index = index - len(ALL_ACTIONS.player_action_set)
            card_index = card_action_index // len(ALL_ACTIONS.card_action_set[0])
            within_card_index = card_action_index % len(ALL_ACTIONS.card_action_set[0])
            return ALL_ACTIONS.card_action_set[card_index][within_card_index]
