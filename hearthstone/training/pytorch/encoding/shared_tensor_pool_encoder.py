import collections
import queue
from typing import List, Union

import torch

from hearthstone.simulator.agent.actions import StandardAction
from hearthstone.simulator.core.player import Player
from hearthstone.training.pytorch.encoding.state_encoding import State, EncodedActionSet, Feature, Encoder, \
    ActionComponent

# A global singleton queue, since multiprocessing can't handle passing the queue through as a function argument.
global_process_tensor_queue: queue.Queue = torch.multiprocessing.Queue()

global_thread_tensor_queue: collections.deque = collections.deque()


class SharedTensorPoolEncoder(Encoder):
    def __init__(self, base_encoder: Encoder, multiprocess: bool):
        self.base_encoder: Encoder = base_encoder
        self.multiprocess: bool = multiprocess
        # Pools of tensors to reuse
        self._states_pool: List[State] = []
        self._valid_actions_pool: List[EncodedActionSet] = []

    def _fill_from_queue(self):
        global global_process_tensor_queue
        global global_thread_tensor_queue
        if self._states_pool and self._valid_actions_pool:
            return
        try:
            if self.multiprocess:
                state, valid_actions = global_process_tensor_queue.get_nowait()
            else:
                state, valid_actions = global_thread_tensor_queue.popleft()
        except queue.Empty:
            return
        except AttributeError:
            return
        except IndexError:
            return
        self._states_pool.append(state)
        self._valid_actions_pool.append(valid_actions)

    def encode_state(self, player: Player) -> State:
        base_state = self.base_encoder.encode_state(player)
        self._fill_from_queue()
        if self._states_pool:
            reused_state = self._states_pool.pop()
            reused_state.player_tensor.copy_(base_state.player_tensor).detach_()
            reused_state.cards_tensor.copy_(base_state.cards_tensor).detach_()
            return reused_state
        else:
            return base_state

    def encode_valid_actions(self, player: Player, rearrange_phase: bool = False) -> EncodedActionSet:
        base_action_set: EncodedActionSet = self.base_encoder.encode_valid_actions(player, rearrange_phase)
        self._fill_from_queue()
        if self._valid_actions_pool:
            reused_action_set = self._valid_actions_pool.pop()
            reused_action_set.player_action_tensor.copy_(base_action_set.player_action_tensor).detach_()
            reused_action_set.card_action_tensor.copy_(base_action_set.card_action_tensor).detach_()
            reused_action_set.rearrange_phase.copy_(base_action_set.rearrange_phase).detach_()
            reused_action_set.cards_to_rearrange.copy_(base_action_set.cards_to_rearrange).detach_()
            return reused_action_set
        else:
            return base_action_set

    def player_encoding(self) -> Feature:
        return self.base_encoder.player_encoding()

    def cards_encoding(self) -> Feature:
        return self.base_encoder.cards_encoding()

    def action_encoding_size(self) -> int:
        return self.base_encoder.action_encoding_size()

    def get_action_component_index(self, action: Union[StandardAction, ActionComponent]) -> int:
        return self.base_encoder.get_action_component_index(action)

    def get_indexed_action_component(self, index: int) -> ActionComponent:
        return self.base_encoder.get_indexed_action_component(index)
