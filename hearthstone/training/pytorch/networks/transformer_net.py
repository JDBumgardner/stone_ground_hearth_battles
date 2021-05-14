from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_

from hearthstone.simulator.agent.actions import Action, RearrangeCardsAction, StandardAction, DiscoverChoiceAction, \
    SummonAction, PlaySpellAction
from hearthstone.simulator.core.player import HandIndex, BoardIndex, StoreIndex, SpellIndex
from hearthstone.training.pytorch.encoding import default_encoder
from hearthstone.training.pytorch.encoding.default_encoder import EncodedActionSet
from hearthstone.training.pytorch.encoding.state_encoding import State, Encoder, InvalidAction, SummonComponent, \
    SpellComponent
from hearthstone.training.pytorch.networks.running_norm import ObservationNormalizer
from hearthstone.training.pytorch.replay import ActorCriticGameStepDebugInfo
from plackett_luce.plackett_luce import PlackettLuce


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

    def forward(self, src, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None):
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
        self.fc_player = nn.Linear(encoding.player_encoding().size()[0], self.width - 3)
        card_encoding_size = encoding.cards_encoding().size()[1]
        if redundant:
            card_encoding_size += encoding.player_encoding().size()[0]
        self.fc_cards = nn.Linear(card_encoding_size, self.width - 3)
        spell_encoding_size = encoding.spells_encoding().size()[1]
        if redundant:
            spell_encoding_size += encoding.player_encoding().size()[0]
        self.fc_spells = nn.Linear(spell_encoding_size, self.width - 3)
        self.encoder_layer = TransformerEncoderPostNormLayer(d_model=width, dim_feedforward=width * 4, nhead=4,
                                                             dropout=0.0, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers,
                                                         norm=LayerNorm(width))
        self._reset_parameters()

    @staticmethod
    def pad_with_one_hot(rep: torch.Tensor) -> torch.Tensor:
        return torch.cat((F.one_hot(torch.tensor(0), 3).unsqueeze(0).unsqueeze(0).expand(*rep.shape[-1], -1), rep), dim=2)

    def forward(self, state: State) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(state, State):
            state = State(state[0], state[1], state[2])
        player_rep = self.fc_player(state.player_tensor).unsqueeze(1)
        card_rep = state.cards_tensor
        if self.redundant:
            # Concatenate the player representation to each card representation.
            card_rep = torch.cat(
                (card_rep, state.player_tensor.unsqueeze(1).expand(-1, state.cards_tensor.size()[1], -1)), dim=2)
        card_rep = self.fc_cards(card_rep)

        spell_rep = state.spells_tensor
        if self.redundant:
            # Concatenate the player representation to each spell representation.
            spell_rep = torch.cat(
                (spell_rep, state.player_tensor.unsqueeze(1).expand(-1, state.spells_tensor.size()[1], -1)), dim=2)
        spell_rep = self.fc_spells(spell_rep)

        # We add an indicator dimension to distinguish the player representation from the card representation.
        player_rep = self.pad_with_one_hot(player_rep)
        card_rep = self.pad_with_one_hot(card_rep)
        spell_rep = self.pad_with_one_hot(spell_rep)

        full_rep = torch.cat((player_rep, card_rep, spell_rep), dim=1).permute(1, 0, 2)
        full_rep: torch.Tensor = self.transformer_encoder(full_rep).permute(1, 0, 2)
        return full_rep[:, 0], full_rep[:, 1:-spell_rep.shape], full_rep[:, -spell_rep.shape:]

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class HearthstoneTransformerNet(nn.Module):
    def __init__(self, encoding: Encoder, hidden_layers=1, hidden_size=16, shared=False, activation_function="gelu",
                 redundant=False, normalize_observations=False, normalization_momentum=0.001):
        super().__init__()

        self.encoding = encoding
        self.player_hidden_size = hidden_size
        self.card_hidden_size = hidden_size
        self.spell_hidden_size = hidden_size
        if hidden_layers == 0:
            # If there are no hidden layers, just connect directly to output layers.
            self.player_hidden_size = encoding.player_encoding().size()[0]
            self.card_hidden_size = encoding.cards_encoding().size()[1]

        self.normalize_observations = normalize_observations
        if normalize_observations:
            self.observation_normalizer = ObservationNormalizer(encoding=encoding, gamma=normalization_momentum)

        self.policy_encoder = TransformerWithContextEncoder(encoding, hidden_size, hidden_layers,
                                                            activation_function, redundant=redundant)
        if shared:
            self.value_encoder = self.policy_encoder
        else:
            self.value_encoder = TransformerWithContextEncoder(encoding, hidden_size,
                                                               hidden_layers, activation_function, redundant=redundant)

        # Output layers
        self.fc_player_policy = nn.Linear(self.player_hidden_size, len(default_encoder.ALL_ACTIONS.player_action_set))
        self.fc_card_policy = nn.Linear(self.card_hidden_size, len(default_encoder.ALL_ACTIONS.card_action_set[1]))
        self.fc_spell_policy = nn.Linear(self.spell_hidden_size, 1)
        self.fc_card_position = nn.Linear(self.card_hidden_size, 1, bias=False)

        nn.init.constant_(self.fc_player_policy.weight, 0)
        nn.init.constant_(self.fc_player_policy.bias, 0)
        nn.init.constant_(self.fc_card_policy.weight, 0)
        nn.init.constant_(self.fc_card_policy.bias, 0)
        nn.init.constant_(self.fc_spell_policy.weight, 0)
        nn.init.constant_(self.fc_spell_policy.bias, 0)
        nn.init.constant_(self.fc_card_position.weight, 0)

        # Additional network for battlecry target selection
        target_selection_width = self.card_hidden_size + 4  # +1 for encoding the card being played, but divisible by 4.
        target_selection_encoder = TransformerEncoderPostNormLayer(d_model=target_selection_width,
                                                                   dim_feedforward=target_selection_width * 4, nhead=4,
                                                                   dropout=0.0, activation=activation_function)
        self.target_selection_transformer = nn.TransformerEncoder(target_selection_encoder, num_layers=1,
                                                                  norm=LayerNorm(target_selection_width))
        self.target_selection_fc = nn.Linear(target_selection_width, 1)

        nn.init.constant_(self.target_selection_fc.weight, 0)
        nn.init.constant_(self.target_selection_fc.bias, 0)

        self.fc_value = nn.Linear(self.player_hidden_size, 1)

    def forward(self, state: State, valid_actions: EncodedActionSet, chosen_actions: Optional[List[Action]]):
        if not isinstance(state, State):
            state = State(state[0], state[1], state[2])
        if not isinstance(valid_actions, EncodedActionSet):
            valid_actions = EncodedActionSet(*valid_actions)

        if self.normalize_observations:
            state = self.observation_normalizer(state)

        policy_encoded_player, policy_encoded_cards, policy_encoded_spells = self.policy_encoder(state)
        value_encoded_player, value_encoded_cards, value_encoded_spells = self.value_encoder(state)

        player_policy = self.fc_player_policy(policy_encoded_player)
        card_policy = self.fc_card_policy(policy_encoded_cards)
        spell_policy = self.fc_spell_policy(policy_encoded_spells)

        # Disable invalid actions with a "masked" softmax
        player_policy = player_policy.masked_fill(valid_actions.player_action_tensor.logical_not(), -1e30)
        card_policy = card_policy.masked_fill(valid_actions.card_action_tensor.logical_not(), -1e30)
        spell_policy = spell_policy.masked_fill(valid_actions.spell_action_tensor.logical_not(), -1e30)

        # Flatten the policy
        policy = torch.cat((player_policy.flatten(1), card_policy.flatten(1), spell_policy.flatten(1)), dim=1)

        # The policy network outputs an array of the log probability of each action component.
        policy = F.log_softmax(policy, dim=1)
        # The value network outputs the linear combination of the representation of the player in the last layer,
        # which will be between -3.5 (8th place) at the minimum and 3.5 (1st place) at the max.
        value = self.fc_value(value_encoded_player).squeeze(1)

        # Distribution over action components, since some actions are more complex than this categorical distribution.
        component_distribution = Categorical(policy.exp())
        if chosen_actions:
            component_samples = torch.tensor(
                [self.encoding.get_action_component_index(action) if isinstance(action,
                                                                                (StandardAction,
                                                                                 DiscoverChoiceAction)) else 0
                 for action in chosen_actions], dtype=torch.int64, device=policy.device)
        else:
            component_samples = component_distribution.sample()
        component_log_probs = component_distribution.log_prob(component_samples)

        # Here we compute the target selection scores for battlecry/magnetic targets
        target_encoded_player = torch.cat(
            (policy_encoded_player,
             torch.zeros((policy_encoded_player.shape[0], 4), device=policy_encoded_player.device)), dim=1)

        sampled_action_mask = torch.zeros_like(policy, dtype=torch.bool).scatter(1, component_samples.unsqueeze(-1), True)
        active_cards = torch.max(torch.reshape(sampled_action_mask[:, player_policy.shape[1]:-spell_policy.shape[1]],
                                               card_policy.shape), dim=2).values
        active_spells = sampled_action_mask[:, -spell_policy.shape[1]:]

        hand_size = valid_actions.battlecry_target_tensor.shape[1]
        board_size = valid_actions.battlecry_target_tensor.shape[2] - 1
        store_size = valid_actions.store_target_spell_action_tensor.shape[2]
        summoned_hand_cards = active_cards[:, valid_actions.hand_start:valid_actions.hand_start + hand_size]
        # TODO: Note that this is only true because Summon is currently the only hand-card action.
        is_targeted_action = torch.max(torch.cat((summoned_hand_cards, active_spells), dim=1), dim=1).values

        target_encoded_cards = torch.cat(
            (policy_encoded_cards, active_cards.unsqueeze(-1).expand(active_cards.shape + (4,))), dim=2)
        target_encoded_spells = torch.cat(
            (policy_encoded_spells, active_spells.unsqueeze(-1).expand(active_spells.shape + (4,))), dim=2)
        target_full_rep = torch.cat((target_encoded_player.unsqueeze(1), target_encoded_cards, target_encoded_spells),
                                    dim=1).permute(1, 0, 2)
        target_full_rep: torch.Tensor = self.target_selection_transformer(target_full_rep).permute(1, 0, 2)

        valid_board_battlecry_no_targets = (valid_actions.no_target_battlecry_tensor * summoned_hand_cards.unsqueeze(-1)).sum(
            dim=1)
        valid_board_battlecry_targets = (valid_actions.battlecry_target_tensor * summoned_hand_cards.unsqueeze(-1)).sum(dim=1)
        valid_spell_no_targets = (valid_actions.no_target_spell_action_tensor * active_spells.unsqueeze(-1)).sum(dim=1)
        valid_store_spell_targets = (valid_actions.store_target_spell_action_tensor * active_spells.unsqueeze(-1)).sum(
            dim=1)
        valid_board_spell_targets = (valid_actions.board_target_spell_action_tensor * active_spells.unsqueeze(-1)).sum(dim=1)

        all_valid_no_targets = torch.logical_or(valid_board_battlecry_no_targets, valid_spell_no_targets)
        all_valid_board_targets = torch.logical_or(valid_board_battlecry_targets, valid_board_spell_targets)

        all_valid_targets = torch.cat((all_valid_no_targets.unsqueeze(-1), valid_store_spell_targets, all_valid_board_targets), dim=1)

        target_no_target_rep = target_full_rep[:, 0:1]
        target_store_rep = target_full_rep[:, 1 + valid_actions.store_start: 1 + valid_actions.store_start + store_size]
        target_board_rep = target_full_rep[:, 1 + valid_actions.board_start: 1 + valid_actions.board_start + board_size]

        target_policy = self.target_selection_fc(torch.cat((target_no_target_rep, target_store_rep, target_board_rep),
                                                           dim=1)).squeeze(-1)
        target_policy = target_policy.masked_fill(all_valid_targets.logical_not(), -1e30)
        target_policy = F.log_softmax(target_policy, dim=1)
        target_distribution = Categorical(target_policy.exp())

        def get_action_target_index(action):
            if isinstance(action, SummonAction) and action.targets:
                return action.targets[0] + 1 + store_size
            elif isinstance(action, PlaySpellAction):
                if action.board_target:
                    return action.board_target + 1 + store_size
                if action.store_target:
                    return action.store_target + 1
            else:
                return 0

        if chosen_actions:
            target_samples = torch.tensor(
                [get_action_target_index(action)for action in chosen_actions], dtype=torch.int, device=policy.device)
        else:
            target_samples = target_distribution.sample()
        target_log_probs = target_distribution.log_prob(target_samples).masked_fill(is_targeted_action.logical_not(), 0.0)

        # We compute a score saying how to order the cards on the board, and use the Plackett Luce distribution to
        # sample permutations.
        card_position_scores = self.fc_card_position(policy_encoded_cards).squeeze(-1)
        card_position_start_index = valid_actions.board_start
        card_position_max_length = int(valid_actions.cards_to_rearrange.max())

        permutation_distribution = PlackettLuce(
            card_position_scores[:, card_position_start_index: card_position_start_index + card_position_max_length],
            valid_actions.cards_to_rearrange * valid_actions.rearrange_phase)
        if chosen_actions:
            permutation_samples = torch.tensor(
                [action.permutation + [0] * (card_position_max_length - len(action.permutation))
                 if isinstance(action, RearrangeCardsAction) else [0] * card_position_max_length
                 for
                 action in chosen_actions], device=card_position_scores.device)
        else:
            permutation_samples = permutation_distribution.sample()
        permutation_log_probs = permutation_distribution.log_prob(permutation_samples)

        output_actions = chosen_actions or [InvalidAction() for _ in range(valid_actions.player_action_tensor.shape[0])]

        # Convert to numpy first for faster random access
        component_samples_numpy = component_samples.detach().cpu().numpy()
        target_samples_numpy = target_samples.detach().cpu().numpy()
        for i in range(valid_actions.player_action_tensor.shape[0]):
            if chosen_actions:
                output_actions[i] = chosen_actions[i]
            else:
                if valid_actions.rearrange_phase[i]:
                    output_actions[i] = RearrangeCardsAction(
                        permutation_samples[i, :valid_actions.cards_to_rearrange[i]].tolist())
                else:
                    output_actions[i] = self.encoding.get_indexed_action_component(component_samples_numpy[i])
                    if isinstance(output_actions[i], SummonComponent):
                        battlecry_targets = [] if target_samples_numpy[i] == 0 else [
                            BoardIndex(target_samples_numpy[i] - 1 - store_size)]
                        output_actions[i] = SummonAction(HandIndex(output_actions[i].index), battlecry_targets)
                    if isinstance(output_actions[i], SpellComponent):
                        store_index = None
                        board_index = None
                        if target_samples_numpy[i] >= 1 + store_size:
                            board_index = BoardIndex(target_samples_numpy[i] - 1 - store_size)
                        elif target_samples_numpy[i] >= 1:
                            store_index = StoreIndex(target_samples_numpy[i])
                        output_actions[i] = PlaySpellAction(SpellIndex(output_actions[i].index),
                                                            board_target=board_index, store_target=store_index)

        action_log_probs = torch.where(valid_actions.rearrange_phase,
                                       permutation_log_probs,
                                       component_log_probs + target_log_probs)
        debug_info = ActorCriticGameStepDebugInfo(
            component_policy=policy,
            permutation_logits=permutation_distribution.logits
        )

        return output_actions, action_log_probs, value, debug_info
