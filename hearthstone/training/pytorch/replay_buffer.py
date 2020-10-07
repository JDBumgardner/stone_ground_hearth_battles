import logging
import random
from typing import List, Generator, Optional

from hearthstone.simulator.replay.replay import Replay
from hearthstone.training.pytorch.normalization import ObservationNormalizer
from hearthstone.training.pytorch.replay import ActorCriticGameStepInfo


class EpochBuffer:
    """
    A replay buffer for a2c or ppo, containing an unordered list of transitions.

    """

    def __init__(self, bot_name: str, observation_normalizer: Optional[ObservationNormalizer] = None):
        """
        :param bot_name:  The name of the agent that this buffer is collecting samples for. Only this bots actions will be added to the replay buffer.
        :param observation_normalizer: Observation normalizer to use for computing rolling average observation normalization.
        """
        self.bot_name = bot_name
        self.transitions: List[ActorCriticGameStepInfo] = []
        self.observation_normalizer = observation_normalizer

    def __len__(self):
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()

    def add_replay(self, replay: Replay):
        for replay_step in replay.steps:
            if replay_step.player == self.bot_name and replay_step.bot_info:
                bot_info = replay_step.bot_info
                if self.observation_normalizer:
                    bot_info = self.observation_normalizer.normalize(bot_info)
                self.transitions.append(bot_info)

    def sample_minibatches(self, batch_size: int) -> Generator[List[ActorCriticGameStepInfo], None, None]:
        random.shuffle(self.transitions)
        i = 0
        while True:
            batch = self.transitions[i:i + batch_size]
            if len(batch) < batch_size:
                break
            yield batch
            i += batch_size
