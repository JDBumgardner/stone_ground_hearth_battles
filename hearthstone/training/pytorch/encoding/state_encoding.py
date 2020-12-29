import copy
from typing import NamedTuple, Any, Tuple, Callable, List

import numpy as np
import torch

from hearthstone.simulator.agent import StandardAction
from hearthstone.simulator.core.cards import MonsterCard, CardLocation
from hearthstone.simulator.core.player import Player


class State(NamedTuple):
    player_tensor: torch.Tensor
    cards_tensor: torch.Tensor

    def to(self, device: torch.device):
        if device:
            return State(self.player_tensor.to(device), self.cards_tensor.to(device))
        else:
            return self


class Transition(NamedTuple):
    state: State
    valid_actions: torch.BoolTensor
    action: int  # Index of the action
    action_prob: float
    value: float
    gae_return: float
    retn: float
    reward: float
    is_terminal: bool


def frozen_player(player: Player) -> Player:
    player = copy.copy(player)
    player.tavern = None
    player = copy.deepcopy(player)
    return player


class LocatedCard:
    def __init__(self, card: MonsterCard, location: CardLocation):
        self.card = card
        self.location = location


class Feature:

    def fill_tensor(self, obj: Any, view: np.ndarray):
        pass

    def size(self) -> Tuple:
        pass

    def dtype(self) -> np.dtype:
        pass

    def encode(self, obj: Any) -> np.ndarray:
        tensor = np.zeros(self.size(), dtype=self.dtype())
        self.fill_tensor(obj, tensor)
        return tensor

    def flattened_size(self) -> int:
        num = 1
        for dim in self.size():
            num *= dim
        return num


class ScalarFeature(Feature):
    def __init__(self, feat: Callable[[Any], Any], dtype=None):
        self._dtype = dtype or np.float32
        self.feat = feat

    def fill_tensor(self, obj: Any, view: np.ndarray):
        view.data[0] = self.feat(obj)

    def size(self) -> Tuple:
        return (1,)

    def dtype(self) -> np.dtype:
        return self._dtype


class OnehotFeature(Feature):
    def __init__(self, extractor: Callable[[Any], Any], num_classes: int, dtype=None):
        self._dtype = dtype or np.float32
        self.extractor = extractor
        self.num_classes = num_classes

    def fill_tensor(self, obj: Any, view: np.ndarray):
        view[self.extractor(obj)] = 1.0

    def size(self) -> Tuple:
        return (self.num_classes,)

    def dtype(self) -> np.dtype:
        return self._dtype


class CombinedFeature(Feature):
    def __init__(self, features: List[Feature], dtype=None):
        self.features = features
        self._dtype = dtype or np.float32

    def fill_tensor(self, obj: Any, view: np.ndarray):
        start = 0
        for feature in self.features:
            size = feature.size()
            feature.fill_tensor(obj, view[start: start + size[0]])
            start += size[0]

    def size(self) -> Tuple:
        sizes = [feature.size() for feature in self.features]
        dimension_sum = 0
        for size in sizes:
            assert size[1:] == sizes[0][1:]
            dimension_sum += size[0]
        return (dimension_sum,) + sizes[0][1:]

    def dtype(self) -> np.dtype:
        return self._dtype


class ListOfFeatures(Feature):
    def __init__(self, extractor: Callable[[Any], List[Any]], feature: Feature, width: int, dtype=None):
        self.extractor = extractor
        self.feature = feature
        self.width = width
        self._dtype = dtype or np.float32

    def fill_tensor(self, obj: Any, view: np.ndarray):
        values_to_encode = self.extractor(obj)
        assert (len(values_to_encode) <= self.width)
        for i, value in enumerate(values_to_encode):
            self.feature.fill_tensor(value, view[i])

    def size(self):
        return (self.width,) + self.feature.size()

    def dtype(self):
        return self._dtype


class SortedByValueFeature(Feature):
    def __init__(self, extractor: Callable[[Any], List[Any]], width: int, dtype=None):
        self.extractor = extractor
        self.width = width
        self._dtype = dtype or np.float32

    def fill_tensor(self, obj: Any, view: np.ndarray):
        values_to_encode = self.extractor(obj)
        sorted_values = sorted(values_to_encode)
        assert (len(values_to_encode) <= self.width)
        for i, value in enumerate(sorted_values):
            view[i] = value

    def size(self):
        return (self.width,)

    def dtype(self):
        return self._dtype


class EncodedActionSet(NamedTuple):
    player_action_tensor: torch.BoolTensor
    card_action_tensor: torch.BoolTensor
    rearrange_phase: torch.BoolTensor
    cards_to_rearrange: torch.IntTensor  # Start and length index

    def to(self, device: torch.device):
        if device:
            return EncodedActionSet(self.player_action_tensor.to(device), self.card_action_tensor.to(device),
                                    self.rearrange_phase.to(device),
                                    self.cards_to_rearrange.to(device))
        else:
            return self

class ActionComponent:
    """
    A component of a potentially multi-part action.
    """
    def valid(self, player: 'Player'):
        raise NotImplementedError()


class ActionSet(NamedTuple):
    player_action_set: List[ActionComponent]
    card_action_set: List[List[ActionComponent]]


class InvalidAction(StandardAction):
    def __repr__(self):
        return f"InvalidAction()"

    def apply(self, player: 'Player'):
        assert False

    def valid(self, player: 'Player') -> bool:
        return False


class Encoder:
    def encode_state(self, player: Player) -> State:
        raise NotImplemented()

    def encode_valid_actions(self, player: Player, rearrange_phase: bool = False) -> EncodedActionSet:
        raise NotImplemented()

    def player_encoding(self) -> Feature:
        raise NotImplemented()

    def cards_encoding(self) -> Feature:
        raise NotImplemented()

    def action_encoding_size(self) -> int:
        raise NotImplemented()

    def get_action_index(self, action: StandardAction) -> int:
        raise NotImplemented()

    def get_indexed_action(self, index: int) -> StandardAction:
        raise NotImplemented()