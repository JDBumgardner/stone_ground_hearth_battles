from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_

from hearthstone.simulator.agent import Action, RearrangeCardsAction, StandardAction, DiscoverChoiceAction
from hearthstone.training.pytorch.encoding import default_encoder
from hearthstone.training.pytorch.encoding.default_encoder import EncodedActionSet
from hearthstone.training.pytorch.encoding.state_encoding import State, Encoder, InvalidAction
from hearthstone.training.pytorch.networks.plackett_luce import PlackettLuce
from hearthstone.training.pytorch.replay import ActorCriticGameStepDebugInfo


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderPostNormLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__()
        assert dropout == 0.0
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        norm_src = self.norm1(src)

        src2 = self.self_attn(norm_src, norm_src, norm_src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + src2

        norm_src = self.norm2(src)
        src2 = self.linear2(self.activation(self.linear1(norm_src)))
        src = src + src2
        return src


class TransformerWithContextEncoder(nn.Module):
    # TODO "redundant" arg should be implemented as a different state encoding instead
    def __init__(self, encoding: Encoder, width: int, num_layers: int, activation: str, redundant=False):
        super().__init__()
        self.redundant = redundant
        self.width = width
        # TODO Orthogonal initialization?
        self.fc_player = nn.Linear(encoding.player_encoding().size()[0], self.width - 1)
        card_encoding_size = encoding.cards_encoding().size()[1]
        if redundant:
            card_encoding_size += encoding.player_encoding().size()[0]
        self.fc_cards = nn.Linear(card_encoding_size, self.width - 1)
        self.encoder_layer = TransformerEncoderPostNormLayer(d_model=width, dim_feedforward=width*4, nhead=4, dropout=0.0, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=LayerNorm(width))
        self._reset_parameters()

    def forward(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        player_rep = self.fc_player(state.player_tensor).unsqueeze(1)
        card_rep = state.cards_tensor
        if self.redundant:
            # Concatenate the player representation to each card representation.
            card_rep = torch.cat(
                (card_rep, state.player_tensor.unsqueeze(1).expand(-1, state.cards_tensor.size()[1], -1)), dim=2)
        card_rep = self.fc_cards(card_rep)

        # We add an indicator dimension to distinguish the player representation from the card representation.
        player_rep = F.pad(player_rep, [1, 0], value=1.0)
        card_rep = F.pad(card_rep, [1, 0], value=0.0)

        full_rep = torch.cat((player_rep, card_rep), dim=1).permute(1, 0, 2)
        full_rep: torch.Tensor = self.transformer_encoder(full_rep).permute(1, 0, 2)
        return full_rep[:, 0], full_rep[:, 1:]

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class HearthstoneTransformerNet(nn.Module):
    def __init__(self, encoding: Encoder, hidden_layers=1, hidden_size=16, shared=False, activation_function="gelu", redundant=False):
        super().__init__()

        self.encoding = encoding
        self.player_hidden_size = hidden_size
        self.card_hidden_size = hidden_size
        if hidden_layers == 0:
            # If there are no hidden layers, just connect directly to output layers.
            self.player_hidden_size = encoding.player_encoding().size()[0]
            self.card_hidden_size = encoding.cards_encoding().size()[1]

        self.policy_encoder = TransformerWithContextEncoder(encoding, hidden_size, hidden_layers,
                                                            activation_function, redundant=redundant)
        if shared:
            self.value_encoder = self.policy_encoder
        else:
            self.value_encoder = TransformerWithContextEncoder(encoding, hidden_size,
                                                               hidden_layers, activation_function, redundant=redundant)

        # Output layers
        self.fc_player_policy = nn.Linear(self.player_hidden_size,
                                          len(default_encoder.ALL_ACTIONS.player_action_set))
        self.fc_card_policy = nn.Linear(self.card_hidden_size,
                                        len(default_encoder.ALL_ACTIONS.card_action_set[1]))
        self.fc_card_position = nn.Linear(self.card_hidden_size, 1)

        nn.init.constant_(self.fc_player_policy.weight, 0)
        nn.init.constant_(self.fc_player_policy.bias, 0)
        nn.init.constant_(self.fc_card_policy.weight, 0)
        nn.init.constant_(self.fc_card_policy.bias, 0)
        nn.init.constant_(self.fc_card_position.weight, 0)
        nn.init.constant_(self.fc_card_position.bias, 0)

        # Additional network for battlecry target selection
        target_selection_width = self.card_hidden_size+4  # +1 for encoding the card being played, but divisible by 4.
        target_selection_encoder = TransformerEncoderPostNormLayer(d_model=target_selection_width, dim_feedforward=target_selection_width*4, nhead=4, dropout=0.0, activation=activation_function)
        self.target_selection_transformer = nn.TransformerEncoder(target_selection_encoder, num_layers=1, norm=LayerNorm(target_selection_width))
        self.target_selection_fc = nn.Linear(target_selection_width, 1)

        nn.init.constant_(self.target_selection_fc.weight, 0)
        nn.init.constant_(self.target_selection_fc.bias, 0)

        self.fc_value = nn.Linear(self.player_hidden_size, 1)

    def forward(self, state: State, valid_actions: EncodedActionSet, chosen_actions: Optional[List[Action]]):
        if not isinstance(state, State):
            state = State(state[0], state[1])
        if not isinstance(valid_actions, EncodedActionSet):
            valid_actions = EncodedActionSet(valid_actions[0], valid_actions[1], valid_actions[2], valid_actions[3])

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
        value = self.fc_value(value_encoded_player).squeeze(1)

        # Distribution over action components, since some actions are more complex than this categorical distribution.
        component_distribution = Categorical(policy.exp())
        if chosen_actions:
            component_samples = torch.tensor(
                [self.encoding.get_action_index(action) if isinstance(action, (StandardAction, DiscoverChoiceAction)) else 0 for action in
                 chosen_actions], dtype=torch.int, device=policy.device)
        else:
            component_samples = component_distribution.sample()
        component_log_probs = component_distribution.log_prob(component_samples)

        # We compute a score saying how to order the cards on the board, and use the Plackett Luce distribution to
        # sample permutations.
        card_position_scores = self.fc_card_position(policy_encoded_cards).squeeze(-1)
        card_position_start_index = int(valid_actions.cards_to_rearrange[:, 0].max())
        card_position_max_length = int(valid_actions.cards_to_rearrange[:, 1].max())
        assert card_position_start_index == int(valid_actions.cards_to_rearrange[:, 0].min())
        permutation_distribution = PlackettLuce(
            card_position_scores[:, card_position_start_index: card_position_start_index + card_position_max_length],
            valid_actions.cards_to_rearrange[:, 1] * valid_actions.rearrange_phase)
        if chosen_actions:
            permutation_samples = torch.nn.utils.rnn.pad_sequence(
                [torch.LongTensor(action.permutation) if isinstance(action,
                                                                    RearrangeCardsAction) else torch.LongTensor() for
                 action in chosen_actions],
                batch_first=True).to(device=card_position_scores.device)
        else:
            permutation_samples = permutation_distribution.sample()
        permutation_log_probs = permutation_distribution.log_prob(permutation_samples)

        output_actions = chosen_actions or [InvalidAction() for _ in range(valid_actions.player_action_tensor.shape[0])]
        action_log_probs = torch.where(valid_actions.rearrange_phase,
                                       permutation_log_probs,
                                       component_log_probs)
        for i in range(valid_actions.player_action_tensor.shape[0]):
            if chosen_actions:
                output_actions[i] = chosen_actions[i]
            else:
                if valid_actions.rearrange_phase[i]:
                    output_actions[i] = RearrangeCardsAction(
                        permutation_samples[i, :valid_actions.cards_to_rearrange[i, 1]].tolist())
                else:
                    output_actions[i] = self.encoding.get_indexed_action(component_samples[i])

        debug_info = ActorCriticGameStepDebugInfo(
            component_policy=policy,
            permutation_logits=permutation_distribution.logits
        )

        return output_actions, action_log_probs, value, debug_info
