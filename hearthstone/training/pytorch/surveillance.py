from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical

from hearthstone.agent import Action
from hearthstone.training.pytorch.hearthstone_state_encoder import State, EncodedActionSet, get_indexed_action, \
    encode_player, encode_valid_actions, Transition
from hearthstone.training.pytorch.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.replay_buffer import ReplayBuffer, logger


class SurveiledPytorchBot(PytorchBot):
    def __init__(self, net: nn.Module, replay_buffer: ReplayBuffer):
        """
        Puts transitions into the replay buffer.

        Args:
            net: Neural net
            replay_buffer: Buffer of transitions.
        """
        super().__init__(net)
        self.replay_buffer = replay_buffer
        self.last_state: Optional[State] = None
        self.last_action: Optional[Action] = None
        self.last_action_prob: Optional[float] = None
        self.last_valid_actions: Optional[EncodedActionSet] = None

    def buy_phase_action(self, player: 'Player') -> Action:
        policy, value = self.policy_and_value(player)
        probs = torch.exp(policy[0])
        action_index = Categorical(probs).sample()
        action = get_indexed_action(int(action_index))
        if not action.valid(player):
            logger.debug("No! Bad Citizen!")
            logger.debug("This IDE is lit")
        else:
            new_state = encode_player(player)
            if self.last_state is not None:
                self.remember_result(new_state, 0, False)
            self.last_state = encode_player(player)
            self.last_valid_actions = encode_valid_actions(player)
            self.last_action = int(action_index)
            self.last_action_prob = float(policy[0][action_index])
            self.last_value = float(value)
        return action

    def game_over(self, player: 'Player', ranking: int):
        if self.last_state is not None:
            self.remember_result(encode_player(player), 3.5 - ranking, True)

    def remember_result(self, new_state, reward, is_terminal):
        self.replay_buffer.push(Transition(self.last_state, self.last_valid_actions,
                                           self.last_action, self.last_action_prob,
                                           self.last_value,
                                           new_state, reward, is_terminal))