from typing import Optional

import torch

from hearthstone.training.pytorch.hearthstone_state_encoder import State


class ActorCriticGameStepInfo:
    """
    Extra information attached to ReplaySteps when training Actor Critic models.

    This records the state of the game, the policy and value nets outputs, the action taken, and the reward received. It
    also optionally contains information propagated about future returns in gae_info.
    """
    state: State
    valid_actions: torch.BoolTensor
    action: int  # Index of the action
    action_prob: float
    value: float
    reward: float
    is_terminal: bool
    gae_info: Optional['GAEReplayInfo']


class GAEReplayInfo:
    """
    Information calculated based on the entire episode, about future returns, to be attached to ReplaySteps.
    """
    gae_return: float
    retrn: float
