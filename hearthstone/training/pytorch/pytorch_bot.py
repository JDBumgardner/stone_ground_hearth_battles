import logging
import random
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.distributions import Categorical

from hearthstone.agent import Agent, Action
from hearthstone.training.pytorch.hearthstone_state_encoder import encode_player, encode_valid_actions, State, \
    EncodedActionSet, get_indexed_action

logger = logging.getLogger(__name__)


class PytorchBot(Agent):
    def __init__(self, net: nn.Module, device: Optional[torch.device] = None):
        self.authors = []
        self.net: nn.Module = net
        self.device = device
        if self.device:
            self.net.to(device)

    def policy_and_value(self, player: 'Player') -> Tuple[Tensor, float]:
        encoded_state: State = encode_player(player, self.device)
        valid_actions_mask: EncodedActionSet = encode_valid_actions(player, self.device)
        policy, value = self.net(State(encoded_state.player_tensor.unsqueeze(0),
                                       encoded_state.cards_tensor.unsqueeze(0)),
                                 EncodedActionSet(valid_actions_mask.player_action_tensor.unsqueeze(0),
                                                  valid_actions_mask.card_action_tensor.unsqueeze(0)))
        return policy, value

    async def buy_phase_action(self, player: 'Player') -> Action:
        policy, _value = self.policy_and_value(player)
        action = Categorical(torch.exp(policy)).sample()
        return get_indexed_action(int(action))

    # TODO handle learning card and discover choice actions
    async def rearrange_cards(self, player: 'Player') -> List['MonsterCard']:
        return player.in_play

    async def discover_choice_action(self, player: 'Player') -> 'MonsterCard':
        return random.choice(player.discover_queue[0])

