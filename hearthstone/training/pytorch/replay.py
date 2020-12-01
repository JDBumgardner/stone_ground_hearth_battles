from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch

from hearthstone.training.pytorch.encoding.default_encoder import EncodedActionSet
from hearthstone.training.pytorch.encoding.state_encoding import State


@dataclass
class ActorCriticGameStepInfo:
    """
    Extra information attached to ReplaySteps when training Actor Critic models.

    This records the state of the game, the policy and value nets outputs, the action taken, and the reward received. It
    also optionally contains information propagated about future returns in gae_info.
    """
    state: State
    valid_actions: EncodedActionSet
    action: int  # Index of the action
    policy: torch.Tensor
    value: float
    gae_info: Optional['GAEReplayInfo']



class GAEReplayInfo(NamedTuple):
    """
    Information calculated based on the entire episode, about future returns, to be attached to ReplaySteps.
    """
    is_terminal: bool
    reward: float
    gae_return: float
    retrn: float
