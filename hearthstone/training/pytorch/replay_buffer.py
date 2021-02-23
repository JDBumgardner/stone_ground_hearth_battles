import collections
import logging
import random
from queue import Queue
from typing import List, Generator, Optional, Union

from hearthstone.simulator.replay.replay import Replay
from hearthstone.training.pytorch.replay import ActorCriticGameStepInfo


class EpochBuffer:
    """
    A replay buffer for a2c or ppo, containing an unordered list of transitions.

    """

    def __init__(self, bot_name: str):
        """
        :param bot_name:  The name of the agent that this buffer is collecting samples for. Only this bots actions will be added to the replay buffer.
        :param observation_normalizer: Observation normalizer to use for computing rolling average observation normalization.
        """
        self.bot_name = bot_name
        self.transitions: List[ActorCriticGameStepInfo] = []

    def __len__(self):
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()

    def recycle(self, queue: Union[Queue, collections.deque]):
        for transition in self.transitions:
            if isinstance(queue, collections.deque):
                queue.append((transition.state, transition.valid_actions))
            else:
                queue.put_nowait((transition.state, transition.valid_actions))
        self.clear()

    def add_replay(self, replay: Replay):
        for replay_step in replay.steps:
            if replay_step.player == self.bot_name and replay_step.agent_annotation:
                bot_info = replay_step.agent_annotation
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
