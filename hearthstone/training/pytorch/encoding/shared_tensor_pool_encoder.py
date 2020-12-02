import queue
from typing import List, Optional

import torch

from hearthstone.simulator.agent import StandardAction
from hearthstone.simulator.core.player import Player
from hearthstone.training.pytorch.encoding.default_encoder import DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING, \
    ALL_ACTIONS, ALL_ACTIONS_DICT
from hearthstone.training.pytorch.encoding.state_encoding import State, EncodedActionSet, Feature, Encoder
import torch.multiprocessing as mp
from queue import Queue
import numpy as np

from hearthstone.training.pytorch.replay import ActorCriticGameStepInfo

# A global singleton queue, since multiprocessing can't handle passing the queue through as a function argument.
global_tensor_queue: queue.Queue = torch.multiprocessing.Queue()

class SharedTensorPoolEncoder(Encoder):
    def __init__(self, base_encoder: Encoder):
        self.base_encoder: Encoder = base_encoder
        # Pools of tensors to reuse
        self._states_pool: List[State] = []
        self._valid_actions_pool: List[EncodedActionSet] = []

    def _fill_from_queue(self):
        global global_tensor_queue
        if self._states_pool and self._valid_actions_pool:
            return
        try:
            game_step_info: ActorCriticGameStepInfo = global_tensor_queue.get_nowait()
        except queue.Empty:
            return
        except AttributeError:
            return
        self._states_pool.append(game_step_info.state)
        self._valid_actions_pool.append(game_step_info.valid_actions)

    def encode_state(self, player: Player) -> State:
        base_state = self.base_encoder.encode_state(player)
        self._fill_from_queue()
        if self._states_pool:
            reused_state = self._states_pool.pop()
            reused_state.player_tensor[:] = base_state.player_tensor
            reused_state.cards_tensor[:] = base_state.cards_tensor
            return reused_state
        else:
            return base_state

    def encode_valid_actions(self, player: Player) -> EncodedActionSet:
        base_action_set: EncodedActionSet = self.base_encoder.encode_valid_actions(player)
        self._fill_from_queue()
        if self._valid_actions_pool:
            reused_action_set = self._valid_actions_pool.pop()
            reused_action_set.player_action_tensor[:] = base_action_set.player_action_tensor
            reused_action_set.card_action_tensor[:] = base_action_set.card_action_tensor
            return reused_action_set
        else:
            return base_action_set

    def player_encoding(self) -> Feature:
        return self.base_encoder.player_encoding()

    def cards_encoding(self) -> Feature:
        return self.base_encoder.cards_encoding()

    def action_encoding_size(self) -> int:
        return self.base_encoder.action_encoding_size()

    def get_action_index(self, action: StandardAction) -> int:
        return self.base_encoder.get_action_index(action)

    def get_indexed_action(self, index: int) -> StandardAction:
        return self.base_encoder.get_indexed_action(index)
