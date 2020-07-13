import torch
from torch import nn

from hearthstone.training.pytorch.hearthstone_state_encoder import action_encoding_size, State, state_encoding_size, \
    Feature
import torch.nn.functional as F


class HearthstoneFFNet(nn.Module):
    def __init__(self, player_encoding: Feature, card_encoding: Feature):
        super(HearthstoneFFNet, self).__init__()
        self.hidden_size = 64
        # Shared hidden layer
        self.fc1 = nn.Linear(player_encoding.flattened_size() + card_encoding.flattened_size(), self.hidden_size)
        self.fc_policy = nn.Linear(self.hidden_size, action_encoding_size())
        self.fc_value = nn.Linear(self.hidden_size, 1)

    def forward(self, state: State, valid_actions: torch.Tensor):
        x = torch.cat(state.player_tensor.flatten(1) + state.cards_tensor.flatten(1))
        x = F.relu(self.fc1(x))
        policy = self.fc_policy(x)
        # Disable invalid actions with a "masked" softmax
        policy = policy.masked_fill(valid_actions.logical_not(), -1e45)
        policy = F.log_softmax(policy)
        value = self.fc_value(x)
        return policy, value
