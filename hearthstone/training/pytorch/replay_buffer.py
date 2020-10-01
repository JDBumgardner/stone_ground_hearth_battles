import logging
import random
from typing import List

from hearthstone.training.pytorch.hearthstone_state_encoder import Transition, State, Feature
from hearthstone.training.pytorch.normalization import PPONormalizer

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        A circular buffer of transitions.

        Args:
            capacity: Size of the circular buffer.
        """
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

    def __init__(self, capacity: int, gamma: float, player_encoding: Feature, cards_encoding: Feature):
        """
        A replay buffer that normalizes the observations before saving them in the replay buffer.

        Args:
            capacity: The size of the buffer
            gamma: The reward discount, which is used for the exponential moving average.
            player_encoding: We need this for the shape of the tensor.
            cards_encoding: We need this for the shape of the tensor.
        """
        super().__init__(capacity)
        self.player_normalizer = PPONormalizer(gamma, player_encoding.size())
        self.cards_normalizer = PPONormalizer(gamma, cards_encoding.size())

    def push(self, transition: Transition):
        # TODO(jeremy): Observations should have 0 mean.
        super().push(Transition(state=State(self.player_normalizer.normalize(transition.state.player_tensor),
                                      self.cards_normalizer.normalize(transition.state.cards_tensor)),
                                valid_actions=transition.valid_actions,
                                action=transition.action,
                                action_prob=transition.action_prob,
                                value=transition.value,
                                value_target=transition.value_target
                                ))


