import collections
from typing import Optional, List

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from hearthstone.simulator.agent import StandardAction
from hearthstone.training.pytorch.hearthstone_state_encoder import State, EncodedActionSet, get_indexed_action, \
    encode_player, encode_valid_actions, Transition, get_action_index
from hearthstone.training.pytorch.pytorch_bot import PytorchBot


class GlobalStepContext:
    def get_global_step(self) -> int:
        raise NotImplemented("Not Implemented")

    def should_plot(self) -> bool:
        raise NotImplemented("Note Implemented")
