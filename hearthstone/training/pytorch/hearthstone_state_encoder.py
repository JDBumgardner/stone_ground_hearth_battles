import copy
import enum
from collections import namedtuple
from typing import Callable, List, Any, Optional

import torch

from hearthstone.agent import TripleRewardsAction, TavernUpgradeAction, RerollAction, \
    EndPhaseAction, SummonAction, BuyAction, SellFromBoardAction, SellFromHandAction, Action
from hearthstone.cards import Card
from hearthstone.monster_types import MONSTER_TYPES
from hearthstone.player import Player, StoreIndex, HandIndex, BoardIndex

State = namedtuple('State', ('player_tensor', 'cards_tensor'))

Transition = namedtuple('Transition',
                        ('state', 'valid_actions', 'action', 'next_state', 'reward'))


def frozen_player(player: Player) -> Player:
    player = copy.copy(player)
    player.tavern = None
    player = copy.deepcopy(player)
    return player


MAX_ENCODED_STORE = 7
MAX_ENCODED_HAND = 10
MAX_ENCODED_BOARD = 7


class CardLocation(enum.Enum):
    STORE = 1
    HAND = 2
    BOARD = 3


class LocatedCard:
    def __init__(self, card: Card, location: CardLocation):
        self.card = card
        self.location = location


class Feature:
    def encode(self, obj: Any) -> torch.Tensor:
        pass

    def size(self) -> torch.Size:
        pass

    def flattened_size(self) -> int:
        num = 1
        for dim in self.size():
            num *= dim
        return num


class ScalarFeature(Feature):
    def __init__(self, feat: Callable[[Any], float]):
        self.feat = feat

    def encode(self, obj: Any) -> torch.Tensor:
        return torch.tensor([self.feat(obj)])

    def size(self) -> torch.Size:
        return torch.Size([1])


class OnehotFeature(Feature):
    def __init__(self, extractor: Callable[[Any], float], num_classes: int):
        self.extractor = extractor
        self.num_classes = num_classes

    def encode(self, card: Any) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.as_tensor(self.extractor(card)), num_classes=self.num_classes).float()

    def size(self) -> torch.Size:
        return torch.Size([self.num_classes])


class CombinedFeature(Feature):
    def __init__(self, features: List[Feature]):
        self.features = features

    def encode(self, card: Any) -> torch.Tensor:
        return torch.cat([feature.encode(card) for feature in self.features])

    def size(self) -> torch.Size:
        sizes = [feature.size() for feature in self.features]
        dimension_sum = 0
        for size in sizes:
            assert size[1:] == sizes[0][1:]
            dimension_sum += size[0]

        return (dimension_sum,) + sizes[0][1:]


class ListOfFeatures(Feature):
    def __init__(self, extractor: Callable[[Any], List[Any]], feature: Feature, width: int, dtype=None):
        self.extractor = extractor
        self.feature = feature
        self.width = width
        self.dtype = dtype or torch.float

    def encode(self, obj: Any) -> torch.Tensor:
        values_to_encode = self.extractor(obj)
        assert(len(values_to_encode) <= self.width)
        padding = [torch.zeros(self.feature.size(), dtype=self.dtype)] * (self.width - len(values_to_encode))
        return torch.stack([self.feature.encode(card) for card in values_to_encode] + padding)

    def size(self):
        return torch.Size((self.width,) + self.feature.size())


def enum_to_int(value: Optional[enum.Enum]) -> int:
    if value is not None:
        return value.value
    else:
        return 0


def default_card_encoding() -> Feature:
    """
    Default encoder for type `LocatedCard`.
    """
    return CombinedFeature([
        ScalarFeature(lambda card: 1.0),  # Present
        ScalarFeature(lambda card: float(card.card.tier)),
        ScalarFeature(lambda card: float(card.card.attack)),
        ScalarFeature(lambda card: float(card.card.health)),
        ScalarFeature(lambda card: float(card.card.taunt)),
        ScalarFeature(lambda card: float(card.card.divine_shield)),
        OnehotFeature(lambda card: enum_to_int(card.card.monster_type), len(MONSTER_TYPES) + 1),
        OnehotFeature(lambda card: enum_to_int(card.location), len(CardLocation) + 1),
    ])


def default_player_encoding() -> Feature:
    """
    Default encoder for the player level features (non-card features).

    Encodes a `Player`.
    """
    return CombinedFeature([ScalarFeature(lambda player: float(player.health)),
                            ScalarFeature(lambda player: float(player.coins)),
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
            default_card_encoding(), MAX_ENCODED_BOARD)
    ])


def encode_player(player: Player) -> State:
    player_tensor = default_player_encoding().encode(player)
    cards_tensor = default_cards_encoding().encode(player)
    return State(player_tensor, cards_tensor)


EncodedActionSet = namedtuple('EncodedActionSet', ('player_action_tensor', 'card_action_tensor'))

ActionSet = namedtuple('ActionSet', ('player_action_set', 'card_action_set'))


class InvalidAction(Action):
    def apply(self, player: 'Player'):
        assert False

    def valid(self, player: 'Player') -> bool:
        return False


def store_indices() -> List[StoreIndex]:
    return [StoreIndex(i) for i in range(MAX_ENCODED_STORE)]


def hand_indices() -> List[HandIndex]:
    return [HandIndex(i) for i in range(MAX_ENCODED_HAND)]


def board_indices() -> List[BoardIndex]:
    return [BoardIndex(i) for i in range(MAX_ENCODED_BOARD)]


def all_actions() -> ActionSet:
    player_action_set = [TripleRewardsAction(), TavernUpgradeAction(), RerollAction(), EndPhaseAction(False),
                         EndPhaseAction(True)]
    store_action_set = [[BuyAction(index), InvalidAction(), InvalidAction()] for index in store_indices()]
    hand_action_set = [[InvalidAction(), SummonAction(index), SellFromHandAction(index)] for index in
                       hand_indices()]
    board_action_set = [[InvalidAction(), InvalidAction(), SellFromBoardAction(index)] for index in
                        board_indices()]
    return ActionSet(player_action_set, store_action_set + hand_action_set + board_action_set)


def encode_valid_actions(player: Player) -> EncodedActionSet:
    actions = all_actions()
    player_action_tensor = torch.tensor([action.valid(player) for action in actions.player_action_set])
    cards_action_tensor = torch.tensor(
        [[action.valid(player) for action in card_actions] for card_actions in actions.card_action_set])
    return EncodedActionSet(player_action_tensor, cards_action_tensor)
