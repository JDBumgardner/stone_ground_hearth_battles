import logging
import random
from typing import Optional, List

import torch
from torch import nn
from torch.distributions import Categorical

from hearthstone.agent import Agent, Action
from hearthstone.training.pytorch.hearthstone_state_encoder import Transition, State, encode_player, \
    encode_valid_actions, EncodedActionSet, get_action_index, get_indexed_action
from hearthstone.training.pytorch.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.normalization import WelfordAggregator, PPONormalizer

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)


class NormalizingReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, gamma, player_encoding, cards_encoding):
        super().__init__(capacity)
        self.player_normalizer = PPONormalizer(gamma, player_encoding.size())
        self.cards_normalizer = PPONormalizer(gamma, cards_encoding.size())

    def push(self, transition: Transition):
        super().push(Transition(state=State(self.player_normalizer.normalize(transition.state.player_tensor),
                                      self.cards_normalizer.normalize(transition.state.cards_tensor)),
                                valid_actions=transition.valid_actions,
                                action=transition.action,
                                action_prob=transition.action_prob,
                                next_state=State(self.player_normalizer.normalize(transition.next_state.player_tensor),
                                                 self.cards_normalizer.normalize(transition.next_state.cards_tensor)),
                                reward=transition.reward,
                                is_terminal=transition.is_terminal
                                ))


class SurveiledPytorchBot(PytorchBot):
    def __init__(self, net: nn.Module, replay_buffer: ReplayBuffer):
        super().__init__(net)
        self.replay_buffer = replay_buffer
        self.last_state: Optional[State] = None
        self.last_action: Optional[Action] = None
        self.last_action_prob: Optional[float] = None
        self.last_valid_actions: Optional[EncodedActionSet] = None

    def buy_phase_action(self, player: 'Player') -> Action:
        policy = self.policy(player)
        action_index = Categorical(torch.exp(policy[0])).sample()
        action = get_indexed_action(int(action_index))
        if not action.valid(player):
            logger.debug("No! Bad Citizen!")
        else:
            new_state = encode_player(player)
            if self.last_state is not None:
                self.remember_result(new_state, 0, False)
            self.last_state = encode_player(player)
            self.last_valid_actions = encode_valid_actions(player)
            self.last_action = int(action_index)
            self.last_action_prob = float(policy[0][action_index])
        return action

    def game_over(self, player: 'Player', ranking: int):
        if self.last_state is not None:
            self.remember_result(encode_player(player), 3.5 - ranking, True)

    def remember_result(self, new_state, reward, is_terminal):
        self.replay_buffer.push(Transition(self.last_state, self.last_valid_actions,
                                           self.last_action, self.last_action_prob,
                                           new_state, reward, is_terminal))


