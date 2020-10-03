from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from hearthstone.training.pytorch import hearthstone_state_encoder
from hearthstone.training.pytorch.hearthstone_state_encoder import Feature, State, EncodedActionSet


class TransformerWithContextEncoder(nn.Module):
    def __init__(self, player_encoding: Feature, card_encoding: Feature, width: int, num_layers: int, activation: str):
        super().__init__()
        self.width = width
        # TODO Orthogonal initialization?
        self.fc_player = nn.Linear(player_encoding.size()[0], self.width - 1)
        self.fc_card = nn.Linear(card_encoding.size()[1], self.width - 1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=width, dim_feedforward=width, nhead=4, dropout=0.0, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        player_rep = self.fc_player(state.player_tensor).unsqueeze(1)
        card_rep = self.fc_card(state.cards_tensor)

        # We add an indicator dimension to distinguish the player representation from the card representation.
        player_rep = F.pad(player_rep, [1, 0], value=1.0)
        card_rep = F.pad(card_rep, [1, 0], value=0.0)

        full_rep = torch.cat((player_rep, card_rep), dim=1)
        full_rep: torch.Tensor = self.transformer_encoder(full_rep)
        return full_rep[:, 0], full_rep[:, 1:]


class HearthstoneTransformerNet(nn.Module):
    def __init__(self, player_encoding: Feature, card_encoding: Feature, hidden_layers=1, hidden_size=16, shared=False, activation_function="gelu"):
        super().__init__()

        self.player_hidden_size = hidden_size
        self.card_hidden_size = hidden_size
        if hidden_layers == 0:
            # If there are no hidden layers, just connect directly to output layers.
            self.player_hidden_size = player_encoding.size()[0]
            self.card_hidden_size = card_encoding.size()[1]

        self.policy_encoder = TransformerWithContextEncoder(player_encoding, card_encoding, hidden_size, hidden_layers,
                                                            activation_function)
        if shared:
            self.value_encoder = self.policy_encoder
        else:
            self.value_encoder = TransformerWithContextEncoder(player_encoding, card_encoding, hidden_size,
                                                               hidden_layers, activation_function)

        # Output layers
        self.fc_player_policy = nn.Linear(self.player_hidden_size,
                                          len(hearthstone_state_encoder.ALL_ACTIONS.player_action_set))
        self.fc_card_policy = nn.Linear(self.card_hidden_size,
                                        len(hearthstone_state_encoder.ALL_ACTIONS.card_action_set[1]))
        nn.init.constant_(self.fc_player_policy.weight, 0)
        nn.init.constant_(self.fc_player_policy.bias, 0)
        nn.init.constant_(self.fc_card_policy.weight, 0)
        nn.init.constant_(self.fc_card_policy.bias, 0)

        self.fc_value = nn.Linear(self.player_hidden_size, 1)


    def forward(self, state: State, valid_actions: EncodedActionSet):
        policy_encoded_player, policy_encoded_cards = self.policy_encoder(state)
        value_encoded_player, value_encoded_cards = self.value_encoder(state)

        player_policy = self.fc_player_policy(policy_encoded_player)
        card_policy = self.fc_card_policy(policy_encoded_cards)

        # Disable invalid actions with a "masked" softmax
        player_policy = player_policy.masked_fill(valid_actions.player_action_tensor.logical_not(), -1e30)
        card_policy = card_policy.masked_fill(valid_actions.card_action_tensor.logical_not(), -1e30)

        # Flatten the policy
        policy = torch.cat((player_policy.flatten(1), card_policy.flatten(1)), dim=1)

        # The policy network outputs an array of the log probability of each action.
        policy = F.log_softmax(policy, dim=1)

        # The value network outputs the linear combination of the representation of the player in the last layer,
        # which will be between -3.5 (8th place) at the minimum and 3.5 (1st place) at the max.
        value = self.fc_value(value_encoded_player)
        return policy, value
