import copy
from typing import NamedTuple, Any, Tuple, Callable, List, Union

import numpy as np
import torch

from hearthstone.simulator.agent.actions import StandardAction, Action
from hearthstone.simulator.core.cards import MonsterCard, CardLocation
from hearthstone.simulator.core.player import Player, BoardIndex, HandIndex, SpellIndex


class State(NamedTuple):
    player_tensor: torch.Tensor
    cards_tensor: torch.Tensor
    spells_tensor: torch.Tensor

    def unsqueeze(self):
        return State(*[tensor.unsqueeze(0) for tensor in self])

    def to(self, device: torch.device):
        if device:
            return State(*[tensor.to(device) for tensor in self])
        else:
            return self

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
        size = self.size()
        dtype = self.dtype()
        tensor = np.zeros(size, dtype)
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
        return self.num_classes,

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
    # Boolean tensor. Dimensions are (batch index, action index)
    player_action_tensor: torch.Tensor
    # Boolean tensor. Dimensions are (batch index, card index, action index)
    card_action_tensor: torch.Tensor
    no_target_battlecry_tensor: torch.Tensor  # Dimensions are (batch index, hand index)
    # Boolean tensor. Dimensions are (batch index, hand index, board index)
    battlecry_target_tensor: torch.Tensor  # Boolean tensor

    spell_action_tensor: torch.Tensor  # Dimensions are (batch index, spell index, placeholder for future index)
    no_target_spell_action_tensor: torch.Tensor  # Dimensions are (batch index, spell index)
    store_target_spell_action_tensor: torch.Tensor  # Dimensions are (batch index, spell index, store index)
    board_target_spell_action_tensor: torch.Tensor  # Dimensions are (batch index, spell index, board index)

    rearrange_phase: torch.Tensor  # Boolean
    cards_to_rearrange: torch.Tensor  # Start and length index as integers

    # TODO: Move these into Encoder instead of EncodedActionSet
    store_start: int
    hand_start: int
    board_start: int

    def unsqueeze(self):
        return EncodedActionSet(
            *[tensor.unsqueeze(0) for tensor in self[:-3]],
            self.store_start,
            self.hand_start,
            self.board_start,
        )

    def to(self, device: torch.device):
        if device:
            return EncodedActionSet(
                *[tensor.to(device) for tensor in self[:-3]],
                self.store_start,
                self.hand_start,
                self.board_start,
            )
        else:
            return self


class ActionComponent:
    """
    A component of a potentially multi-part action.
    """

    def valid(self, player: 'Player'):
        raise NotImplementedError()


class SummonComponent(ActionComponent):
    def __init__(self, index: HandIndex):
        self.index = index

    def valid(self, player: 'Player'):
        return player.room_to_summon(self.index)

    def __repr__(self):
        return f"Summon({self.index}, ?)"


class SpellComponent(ActionComponent):
    def __init__(self, index: SpellIndex):
        self.index = index

    def valid(self, player: 'Player'):
        return player.valid_standard_action() and player.spell_can_be_played(self.index)

    def __repr__(self):
        return f"PlaySpell({self.index}, ?)"


class ActionSet(NamedTuple):
    player_action_set: List[ActionComponent]
    card_action_set: List[List[ActionComponent]]
    # First dimension is card played, second dimension is card targeted (with 0 index meaning no card).
    battlecry_no_target_action_set: List[ActionComponent]
    battlecry_action_set: List[List[ActionComponent]]
    spell_action_set: List[ActionComponent]
    no_target_spell_action_set: List[ActionComponent]
    spell_store_action_set: List[List[ActionComponent]]
    spell_board_action_set: List[List[ActionComponent]]


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

    def spells_encoding(self) -> Feature:
        raise NotImplemented()

    def action_encoding_size(self) -> int:
        raise NotImplemented()

    def get_action_component_index(self, action: Union[ActionComponent, Action]) -> int:
        raise NotImplemented()

    def get_indexed_action_component(self, index: int) -> ActionComponent:
        raise NotImplemented()
