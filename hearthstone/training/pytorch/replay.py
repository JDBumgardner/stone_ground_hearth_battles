from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch

from hearthstone.simulator.agent.actions import Action
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
    action: Action
    action_log_prob: float
    value: float
    gae_info: Optional['GAEReplayInfo']
    debug: Optional['ActorCriticGamestepDebugInfo']


@dataclass
class ActorCriticGameStepDebugInfo:
    """
    Additional data recorded for debugging purposes, but not necessary for training.
    """
    component_policy: torch.Tensor
    permutation_logits: torch.Tensor


class GAEReplayInfo(NamedTuple):
    """
    Information calculated based on the entire episode, about future returns, to be attached to ReplaySteps.
    """
    is_terminal: bool
    reward: float
    gae_return: float
    retrn: float
