import torch
from torch import nn

from hearthstone.training.pytorch.hearthstone_state_encoder import action_encoding_size, State, state_encoding_size
import torch.nn.functional as F


class HearthstoneFFNet(nn.Module):
    def __init__(self):
        super(HearthstoneFFNet, self).__init__()
        self.fc1 = nn.Linear(state_encoding_size(), 64)
        self.fc_policy = nn.Linear(64, action_encoding_size())
        self.fc_value = nn.Linear(64, 1)

    def forward(self, state: State, valid_actions: torch.Tensor):
        x = torch.cat(state.player_tensor.flatten(1) + state.cards_tensor.flatten(1))
        x = F.relu(self.fc1(x))
        policy = self.fc_policy(x)
        # Disable invalid actions with a "masked" softmax
        policy = policy.masked_fill(valid_actions, 1e-45)
        policy = F.log_softmax(policy)
        value = self.fc_value(x)
        return policy, value
