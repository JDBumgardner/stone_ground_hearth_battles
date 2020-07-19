from collections import namedtuple
from typing import List

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from hearthstone.battlebots.cheapo_bot import CheapoBot
from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.battlebots.priority_bot import PriorityBot
from hearthstone.battlebots.priority_functions import attack_health_priority_bot
from hearthstone.battlebots.random_bot import RandomBot
from hearthstone.battlebots.saurolisk_bot import SauroliskBot
from hearthstone.battlebots.supremacy_bot import SupremacyBot

from hearthstone.hero import EmptyHero
from hearthstone.ladder.ladder import Contestant
from hearthstone.monster_types import MONSTER_TYPES
from hearthstone.tavern import Tavern
from hearthstone.training.pytorch.hearthstone_state_encoder import encode_player, encode_valid_actions, Transition, \
    EncodedActionSet


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
    all_bots = [Contestant(f"RandomBot {i}", lambda: RandomBot(i)) for i in range(20)]
    all_bots += [Contestant(f"NoActionBot ", lambda: NoActionBot())]
    all_bots += [Contestant(f"CheapoBot", lambda: CheapoBot(3))]
    return all_bots


def easier_contestants():
    all_bots = [Contestant(f"RandomBot {i}", lambda: RandomBot(i)) for i in range(20)]
    all_bots += [Contestant(f"NoActionBot ", lambda: NoActionBot())]
    all_bots += [Contestant(f"CheapoBot", lambda: CheapoBot(3))]
    all_bots += [Contestant(f"SupremacyBot {t}", lambda: SupremacyBot(t, False, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant(f"SupremacyUpgradeBot {t}", lambda: SupremacyBot(t, True, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant("SauroliskBot", lambda: SauroliskBot(5))]
    return all_bots


def easy_contestants():
    all_bots = [Contestant(f"RandomBot",lambda: RandomBot(1))]
    all_bots += [Contestant(f"NoActionBot ", lambda: NoActionBot())]
    all_bots += [Contestant(f"CheapoBot", lambda: CheapoBot(3))]
    all_bots += [Contestant(f"SupremacyBot {t}", lambda: SupremacyBot(t, False, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant(f"SupremacyUpgradeBot {t}", lambda: SupremacyBot(t, True, i)) for i, t in
                 enumerate([MONSTER_TYPES.MURLOC, MONSTER_TYPES.BEAST, MONSTER_TYPES.MECH, MONSTER_TYPES.DRAGON,
                            MONSTER_TYPES.DEMON, MONSTER_TYPES.PIRATE])]
    all_bots += [Contestant("SauroliskBot", lambda: SauroliskBot(5))]
    all_bots += [Contestant("PriorityHealthAttackBot",lambda:  attack_health_priority_bot(10, PriorityBot))]
    return all_bots

StateBatch = namedtuple('StateBatch', ('player_tensor', 'cards_tensor'))
TransitionBatch = namedtuple('TransitionBatch', ('state', 'valid_actions', 'action', 'action_prob',  'next_state', 'reward', 'is_terminal'))


# TODO: Delete all of this
def tensorize_batch(transitions: List[Transition]) -> TransitionBatch:
    player_tensor = torch.stack([transition.state.player_tensor for transition in transitions])
    cards_tensor = torch.stack([transition.state.cards_tensor for transition in transitions])
    valid_player_actions_tensor = torch.stack([transition.valid_actions.player_action_tensor for transition in transitions])
    valid_card_actions_tensor = torch.stack([transition.valid_actions.card_action_tensor for transition in transitions])
    action_tensor = torch.tensor([transition.action for transition in transitions])
    action_prob_tensor = torch.tensor([transition.action_prob for transition in transitions])
    next_player_tensor = torch.stack([transition.next_state.player_tensor for transition in transitions])
    next_cards_tensor = torch.stack([transition.next_state.cards_tensor for transition in transitions])
    reward_tensor = torch.tensor([transition.reward for transition in transitions])
    is_terminal_tensor = torch.tensor([transition.is_terminal for transition in transitions])
    return TransitionBatch(StateBatch(player_tensor, cards_tensor),
                           EncodedActionSet(valid_player_actions_tensor, valid_card_actions_tensor),
                           action_tensor,
                           action_prob_tensor,
                           StateBatch(next_player_tensor, next_cards_tensor),
                           reward_tensor,
                           is_terminal_tensor)
