from typing import List, NamedTuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots.priority_functions import PriorityFunctions
from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.battlebots.saurolisk_bot import SauroliskBot
from hearthstone.battlebots.supremacy_bot import SupremacyBot
from hearthstone.ladder.ladder import Contestant, ContestantAgentGenerator
from hearthstone.simulator.agent.actions import Action
from hearthstone.simulator.core.hero import EmptyHero
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.training.pytorch.encoding.default_encoder import \
    EncodedActionSet
from hearthstone.training.pytorch.replay import ActorCriticGameStepInfo


def add_net_to_tensorboard(tensorboard: SummaryWriter, net: nn.Module):
    tavern = Tavern()
    player = None
    for i in range(8):
        player = tavern.add_player_with_hero(f"player_{i}", EmptyHero())
    tavern.buying_step()
    state = encode_player(player)
    valid_actions = encode_valid_actions(player)
    tensorboard.add_graph(net, (state, valid_actions))


def easiest_contestants():
    all_bots = [Contestant(f"RandomBot {i}", ContestantAgentGenerator(RandomBot, i)) for i in range(20)]
    all_bots += [Contestant(f"NoActionBot ", ContestantAgentGenerator(NoActionBot))]
    all_bots += [Contestant(f"CheapoBot", ContestantAgentGenerator(CheapoBot, 3))]
    return all_bots


def easier_contestants():
    all_bots = [Contestant(f"RandomBot {i}", ContestantAgentGenerator(RandomBot, i)) for i in range(20)]
    all_bots += [Contestant(f"NoActionBot ", ContestantAgentGenerator(NoActionBot))]
    all_bots += [Contestant(f"CheapoBot", ContestantAgentGenerator(CheapoBot, 3))]
    all_bots += [Contestant(f"SupremacyBot {t}", ContestantAgentGenerator(SupremacyBot, t, False, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant(f"SupremacyUpgradeBot {t}", ContestantAgentGenerator(SupremacyBot, t, True, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant("SauroliskBot", ContestantAgentGenerator(SauroliskBot, 5))]
    return all_bots


def easy_contestants():
    all_bots = [Contestant(f"RandomBot", ContestantAgentGenerator(RandomBot, 1))]
    all_bots += [Contestant(f"NoActionBot ", ContestantAgentGenerator(NoActionBot))]
    all_bots += [Contestant(f"CheapoBot", ContestantAgentGenerator(CheapoBot, 3))]
    all_bots += [Contestant(f"SupremacyBot {t}", ContestantAgentGenerator(SupremacyBot, t, False, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant(f"SupremacyUpgradeBot {t}", ContestantAgentGenerator(SupremacyBot, t, True, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant("SauroliskBot", ContestantAgentGenerator(SauroliskBot, 5))]
    all_bots += [
        Contestant("PriorityHealthAttackBot", lambda: PriorityFunctions.attack_health_priority_bot(10, PriorityBot))]
    return all_bots


class StateBatch(NamedTuple):
    player_tensor: torch.Tensor
    cards_tensor: torch.Tensor


class TransitionBatch(NamedTuple):
    state: StateBatch
    valid_actions: EncodedActionSet
    action: List[Action]
    action_log_prob: torch.Tensor
    value: torch.Tensor
    gae_return: torch.Tensor
    retn: torch.Tensor
    reward: torch.Tensor
    is_terminal: torch.Tensor  # Boolean
    debug_component_policy: torch.Tensor


# TODO: Delete all of this
def tensorize_batch(transitions: List[ActorCriticGameStepInfo], device: torch.device) -> TransitionBatch:
    player_tensor = torch.stack([transition.state.player_tensor for transition in transitions]).detach()
    cards_tensor = torch.stack([transition.state.cards_tensor for transition in transitions]).detach()
    valid_player_actions_tensor = torch.stack(
        [transition.valid_actions.player_action_tensor for transition in transitions]).detach()
    valid_card_actions_tensor = torch.stack(
        [transition.valid_actions.card_action_tensor for transition in transitions]).detach()
    rearrange_phase = torch.stack([transition.valid_actions.rearrange_phase for transition in transitions]).detach()
    cards_to_rearrange = torch.stack(
        [transition.valid_actions.cards_to_rearrange for transition in transitions]).detach()
    action_list = [transition.action for transition in transitions]
    action_log_prob_tensor = torch.tensor([transition.action_log_prob for transition in transitions])
    value_tensor = torch.tensor([transition.value for transition in transitions])
    reward_tensor = torch.tensor([transition.gae_info.reward for transition in transitions])
    is_terminal_tensor = torch.tensor([transition.gae_info.is_terminal for transition in transitions])
    gae_return_tensor = torch.tensor([transition.gae_info.gae_return for transition in transitions])
    retrn_tensor = torch.tensor([transition.gae_info.retrn for transition in transitions])
    debug_component_policy_tensor = torch.cat([transition.debug.component_policy for transition in transitions],
                                              dim=0).detach()

    return TransitionBatch(StateBatch(player_tensor.to(device), cards_tensor.to(device)),
                           EncodedActionSet(valid_player_actions_tensor.to(device),
                                            valid_card_actions_tensor.to(device),
                                            rearrange_phase.to(device),
                                            cards_to_rearrange.to(device)),
                           action_list,
                           action_log_prob_tensor.to(device),
                           value_tensor.to(device),
                           gae_return_tensor.to(device),
                           retrn_tensor.to(device),
                           reward_tensor.to(device),
                           is_terminal_tensor.to(device),
                           debug_component_policy_tensor.to(device),
                           )
