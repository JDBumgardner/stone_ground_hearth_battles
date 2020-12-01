import torch
from torch import nn

from hearthstone.training.pytorch.encoding.default_encoder import EncodedActionSet
from hearthstone.training.pytorch.encoding.state_encoding import State, Feature, Encoder
import torch.nn.functional as F


class HearthstoneFFNet(nn.Module):
    def __init__(self, encoding: Encoder, hidden_layers=1, hidden_size=1024, shared=False, activation_function="gelu"):
        ''' This is a generic, fully connected feed-forward neural net.

           This is a pytorch module: https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module

           Args:
             encoding (Encoder): [link to doc]
             hidden_layers (int): The number of hidden layers.
             hidden_size (int): The width of the hidden layers.
             shared (bool): Whether the policy and value NNs share the same weights in the hidden layers.
             activation_function (string): The activation function between layers.
        '''
        super().__init__()
        input_size = encoding.player_encoding().flattened_size() + encoding.cards_encoding().flattened_size()
        if hidden_layers == 0:
            # If there are no hidden layers, just connect directly to output layers.
            hidden_size = input_size
        self.activation_function = activation_function
        self.shared = shared
        self.hidden_layers = hidden_layers
        self.policy_hidden_layers = []
        self.value_hidden_layers = []
        for i in range(hidden_layers):
            self.policy_hidden_layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            nn.init.orthogonal_(self.policy_hidden_layers[-1].weight)
            if shared:
                self.value_hidden_layers.append(self.policy_hidden_layers[-1])
            else:
                # Create new hidden layers for the value network.
                self.value_hidden_layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
                nn.init.orthogonal_(self.value_hidden_layers[-1].weight)

        # Output layers
        self.fc_policy = nn.Linear(hidden_size, encoding.action_encoding_size())
        nn.init.constant_(self.fc_policy.weight, 0)
        nn.init.constant_(self.fc_policy.bias, 0)
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
        # Because we have a fully connected NN, we can just flatten the input tensors.
        x_policy = torch.cat((state.player_tensor.flatten(1), state.cards_tensor.flatten(1)), dim=1)
        # The value network shares the input layer (for now)
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
        
        # The policy network outputs an array of the log probability of each action.
        policy = F.log_softmax(policy, dim=1)
        # The value network outputs the linear combination of the last hidden layer. The value layer predicts the total reward at the end of the game,
        # which will be between -3.5 (8th place) at the minimum and 3.5 (1st place) at the max. 
        value = self.fc_value(x_value).squeeze(1)
        return policy, value
