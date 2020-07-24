import torch
from torch import nn

from hearthstone.training.pytorch.hearthstone_state_encoder import action_encoding_size, State, \
    Feature, EncodedActionSet
import torch.nn.functional as F


class HearthstoneFFNet(nn.Module):
    def __init__(self, player_encoding: Feature, card_encoding: Feature, hidden_layers=1, hidden_size=1024, shared=False, activation_function="gelu"):
        super().__init__()
        input_size = player_encoding.flattened_size() + card_encoding.flattened_size()
        if hidden_layers == 0:
            hidden_size = input_size
        self.activation_function = activation_function
        self.shared = shared
        self.hidden_layers = hidden_layers
        self.policy_hidden_layers = []
        self.value_hidden_layers = []
        for i in range(hidden_layers):
            self.policy_hidden_layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            if shared:
                self.value_hidden_layers.append(self.policy_hidden_layers[-1])
            else:
                self.value_hidden_layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        self.fc_policy = nn.Linear(hidden_size, action_encoding_size())
        self.fc_value = nn.Linear(hidden_size, 1)

    def activation(self, x):
        if self.activation_function == "relu":
            return F.relu(x)
        elif self.activation_function == "gelu":
            return F.gelu(x)
        elif self.activation_function == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation_function == "tanh":
            return torch.tanh(x)

    def forward(self, state: State, valid_actions: EncodedActionSet):
        x_policy = torch.cat((state.player_tensor.flatten(1), state.cards_tensor.flatten(1)), dim=1)
        x_value = x_policy

        for i in range(self.hidden_layers):
            x_policy = self.activation(self.policy_hidden_layers[i](x_policy))
            if self.shared:
                x_value = x_policy
            else:
                x_value = self.activation(
                    self.value_hidden_layers[i](x_value))
        policy = self.fc_policy(x_policy)
        # Disable invalid actions with a "masked" softmax
        valid_action_tensor = torch.cat((valid_actions.player_action_tensor.flatten(1), valid_actions.card_action_tensor.flatten(1)), dim=1)
        policy = policy.masked_fill(valid_action_tensor.logical_not(), -1e30)
        policy = F.log_softmax(policy, dim=1)
        value = self.fc_value(x_value)
        return policy, value
