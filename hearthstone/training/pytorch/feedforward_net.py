import torch
from torch import nn

from hearthstone.training.pytorch.hearthstone_state_encoder import action_encoding_size, State, \
    Feature, EncodedActionSet
import torch.nn.functional as F


class HearthstoneLinearNet(nn.Module):
    def __init__(self, player_encoding: Feature, card_encoding: Feature):
        super().__init__()
        self.hidden_size = 64
        # Shared hidden layer
        self.fc_policy = nn.Linear(player_encoding.flattened_size() + card_encoding.flattened_size(), action_encoding_size())
        self.fc_value = nn.Linear(player_encoding.flattened_size() + card_encoding.flattened_size(), 1)

    def forward(self, state: State, valid_actions: EncodedActionSet):
        x = torch.cat((state.player_tensor.flatten(1), state.cards_tensor.flatten(1)), dim=1)
        policy = self.fc_policy(x)
        # Disable invalid actions with a "masked" softmax
        valid_action_tensor = torch.cat((valid_actions.player_action_tensor.flatten(1),
                                         valid_actions.card_action_tensor.flatten(1)), dim=1)
        policy = policy.masked_fill(valid_action_tensor.logical_not(), -1e30)
        policy = F.log_softmax(policy, dim=1)
        value = self.fc_value(x)
        return policy, value


class HearthstoneFFSharedNet(nn.Module):
    def __init__(self, player_encoding: Feature, card_encoding: Feature):
        super().__init__()
        self.hidden_size = 1024
        # Shared hidden layer
        self.fc_hidden = nn.Linear(player_encoding.flattened_size() + card_encoding.flattened_size(), self.hidden_size)
        self.fc_policy = nn.Linear(self.hidden_size, action_encoding_size())
        self.fc_value = nn.Linear(self.hidden_size, 1)

    def forward(self, state: State, valid_actions: EncodedActionSet):
        x = torch.cat((state.player_tensor.flatten(1), state.cards_tensor.flatten(1)), dim=1)
        hidden = F.relu(self.fc_hidden(x))
        policy = self.fc_policy(hidden)
        # Disable invalid actions with a "masked" softmax
        valid_action_tensor = torch.cat(
            (valid_actions.player_action_tensor.flatten(1), valid_actions.card_action_tensor.flatten(1)), dim=1)
        policy = policy.masked_fill(valid_action_tensor.logical_not(), -1e30)
        policy = F.log_softmax(policy, dim=1)
        value = self.fc_value(hidden)
        return policy, value


class HearthstoneFFNet(nn.Module):
    def __init__(self, player_encoding: Feature, card_encoding: Feature):
        super(HearthstoneFFNet, self).__init__()
        self.hidden_size = 64
        # Shared hidden layer
        self.fc1_policy = nn.Linear(player_encoding.flattened_size() + card_encoding.flattened_size(), self.hidden_size)
        self.fc1_value = nn.Linear(player_encoding.flattened_size() + card_encoding.flattened_size(), self.hidden_size)
        self.fc_policy = nn.Linear(self.hidden_size, action_encoding_size())
        self.fc_value = nn.Linear(self.hidden_size, 1)

    def forward(self, state: State, valid_actions: EncodedActionSet):
        x = torch.cat((state.player_tensor.flatten(1), state.cards_tensor.flatten(1)), dim=1)
        policy_hidden = F.relu(self.fc1_policy(x))
        value_hidden = F.relu(self.fc1_value(x))
        policy = self.fc_policy(policy_hidden)
        # Disable invalid actions with a "masked" softmax
        valid_action_tensor = torch.cat((valid_actions.player_action_tensor.flatten(1), valid_actions.card_action_tensor.flatten(1)), dim=1)
        policy = policy.masked_fill(valid_action_tensor.logical_not(), -1e30)
        policy = F.log_softmax(policy, dim=1)
        value = self.fc_value(value_hidden)
        return policy, value
