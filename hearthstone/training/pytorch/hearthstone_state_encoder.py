import copy
import enum
from collections import namedtuple
from typing import Optional

import torch

from hearthstone.agent import TripleRewardsAction, HeroPowerAction, TavernUpgradeAction, RerollAction, \
    EndPhaseAction, SummonAction
from hearthstone.cards import MonsterCard
from hearthstone.monster_types import MONSTER_TYPES
from hearthstone.player import Player

State = namedtuple('State', ('player_tensor', 'cards_tensor'))

Transition = namedtuple('Transition',
                        ('state', 'valid_actions', 'action', 'next_state', 'reward'))


def frozen_player(player: Player) -> Player:
    player = copy.copy(player)
    player.tavern = None
    player = copy.deepcopy(player)
    return player

MAX_ENCODED_STORE = 7
MAX_ENCODED_HAND = 10
MAX_ENCODED_BOARD = 7


def encode_monster_type(monster_type: Optional[MONSTER_TYPES]) -> torch.Tensor:
    value = 0 if monster_type is None else monster_type.value
    return torch.nn.functional.one_hot(torch.as_tensor(value), num_classes=len(MONSTER_TYPES)+1)


class CardLocation(enum.Enum):
    STORE = 1
    HAND = 2
    BOARD = 3


def state_encoding_size():
    return 2 + card_encoding_size() * (7+10+7)


def card_encoding_size():
    return 5 + len(CardLocation)+1 + len(MONSTER_TYPES)+1


def encode_card(location: CardLocation, card: MonsterCard) -> torch.Tensor:
    location_tensor = torch.nn.functional.one_hot(torch.as_tensor(location.value), num_classes=len(CardLocation)+1)
    monster_type_tensor = encode_monster_type(card.monster_type)
    return torch.cat([torch.tensor([card.tier, card.attack, card.health, card.taunt, card.divine_shield]),
                      location_tensor,
                      monster_type_tensor
                      ], dim=0).float()


def encode_player(player: Player) -> State:
    player_tensor = torch.tensor([player.health, player.coins])
    store_padding = [torch.zeros(card_encoding_size())] * (MAX_ENCODED_STORE - len(player.store))
    store_tensors = [encode_card(CardLocation.STORE, card) for card in player.store] + store_padding
    hand_padding = [torch.zeros(card_encoding_size())] * (MAX_ENCODED_HAND - len(player.hand))
    hand_tensors = [encode_card(CardLocation.HAND, card) for card in player.hand] + hand_padding
    board_padding = [torch.zeros(card_encoding_size())] * (MAX_ENCODED_BOARD - len(player.in_play))
    board_tensors = [encode_card(CardLocation.BOARD, card) for card in player.in_play] + board_padding
    cards_tensor = torch.stack(store_tensors + hand_tensors + board_tensors)
    return State(player_tensor, cards_tensor)


EncodedActionSet = namedtuple('EncodedActionSet', ('player_action_tensor', 'card_action_tensor'))


def action_tensor(buy: bool, summon:bool, sell: bool) -> torch.Tensor:
    return torch.as_tensor([buy, summon, sell])


def action_encoding_size():
    return 6 + 3 * (7+10+7)


def encode_valid_actions(player: Player) -> EncodedActionSet:
    player_action_tensor = torch.as_tensor([TripleRewardsAction().valid(player),
                                            HeroPowerAction().valid(player),
                                            TavernUpgradeAction().valid(player),
                                            RerollAction().valid(player),
                                            EndPhaseAction(True).valid(player),
                                            EndPhaseAction(False).valid(player),
                                            ])
    store_actions = [action_tensor(True, False, False) for card in player.store] + \
                    [action_tensor(False, False, False) for _ in range(MAX_ENCODED_STORE - len(player.store))]
    hand_actions = [action_tensor(False, SummonAction(card, []).valid(player), True) for card in player.store] \
                   + [action_tensor(False, False, False) for _ in range(MAX_ENCODED_HAND - len(player.hand))]
    board_actions = [action_tensor(False, False, True) for card in player.in_play] \
                    + [action_tensor(False, False, False) for _ in range(MAX_ENCODED_BOARD - len(player.in_play))]
    card_actions_tensor = torch.stack(store_actions + hand_actions + board_actions)
    return EncodedActionSet(player_action_tensor, card_actions_tensor)
