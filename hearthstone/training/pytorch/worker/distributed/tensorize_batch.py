from typing import Tuple, List, Optional

import torch

from hearthstone.simulator.agent.actions import Action
from hearthstone.training.common.state_encoding import EncodedActionSet, State
from hearthstone.training.pytorch.replay import ActorCriticGameStepDebugInfo


def _tensorize_batch(batch: List[Tuple[State, EncodedActionSet, Optional[List[Action]]]],
                     device: torch.device) -> Tuple[
    State, EncodedActionSet, Optional[List[Action]]]:
    player_tensor = torch.cat([b[0].player_tensor for b in batch], dim=0).detach()
    cards_tensor = torch.cat([b[0].cards_tensor for b in batch], dim=0).detach()
    spells_tensor = torch.cat([b[0].spells_tensor for b in batch], dim=0).detach()
    valid_player_actions_tensor = torch.cat(
        [b[1].player_action_tensor for b in batch], dim=0).detach()
    valid_card_actions_tensor = torch.cat(
        [b[1].card_action_tensor for b in batch], dim=0).detach()
    valid_battlecry_target_tensor = torch.cat(
        [b[1].battlecry_target_tensor for b in batch], dim=0).detach()
    rearrange_phase = torch.cat([b[1].rearrange_phase for b in batch], dim=0).detach()
    cards_to_rearrange = torch.cat(
        [b[1].cards_to_rearrange for b in batch], dim=0).detach()
    chosen_actions = None if batch[0][2] is None else [b[2] for b in batch]
    return (State(player_tensor=player_tensor.to(device),
                       cards_tensor=cards_tensor.to(device),
                       spells_tensor=spells_tensor.to(device)),
            EncodedActionSet(player_action_tensor=valid_player_actions_tensor,
                             card_action_tensor=valid_card_actions_tensor,
                             battlecry_target_tensor=valid_battlecry_target_tensor,
                             rearrange_phase=rearrange_phase,
                             cards_to_rearrange=cards_to_rearrange,
                             store_start=batch[0][1].store_start,
                             hand_start=batch[0][1].hand_start,
                             board_start=batch[0][1].board_start
                             ).to(device),
            chosen_actions,
            )


def _untensorize_batch(batch_args: List[Tuple[State, EncodedActionSet, Optional[List[Action]]]],
                       output_actions: List[Action], action_log_probs: torch.Tensor, value: torch.Tensor,
                       debug_info: ActorCriticGameStepDebugInfo, device: torch.device) -> List[
    Tuple[List[Action], torch.Tensor, torch.Tensor, ActorCriticGameStepDebugInfo]]:
    result = []
    i = 0
    for (player_state_tensor, _), _, _ in batch_args:
        batch_entry_size = player_state_tensor.shape[0]
        result.append((output_actions[i:i + batch_entry_size],
                       action_log_probs[i:i + batch_entry_size].detach().to(device),
                       value[i:i + batch_entry_size].detach().to(device),
                       ActorCriticGameStepDebugInfo(
                           component_policy=debug_info.component_policy[i:i + batch_entry_size].detach().to(device),
                           permutation_logits=debug_info.permutation_logits[i:i + batch_entry_size].detach().to(device),
                       )
                       ))
        i += batch_entry_size

    return result
