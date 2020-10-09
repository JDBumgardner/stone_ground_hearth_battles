from typing import Optional, NamedTuple

import torch

from hearthstone.training.pytorch.hearthstone_state_encoder import State, EncodedActionSet


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
    action_prob: float
    value: float
    gae_info: Optional['GAEReplayInfo']

    def __init__(self, state: State, valid_actions: EncodedActionSet, action: int, policy: torch.Tensor, value: float,
                 gae_info: Optional['GAEReplayInfo']):
        self.state = state
        self.valid_actions = valid_actions
        self.action = action
        self.policy = policy
        self.value = value
        self.gae_info = gae_info


class GAEReplayInfo(NamedTuple):
    """
    Information calculated based on the entire episode, about future returns, to be attached to ReplaySteps.
    """
    is_terminal: bool
    reward: float
    gae_return: float
    retrn: float
