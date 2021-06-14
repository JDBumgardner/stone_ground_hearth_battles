import enum
from typing import List, Optional, Dict, Union

import numpy as np
import torch

from hearthstone.simulator.agent.actions import TavernUpgradeAction, RerollAction, EndPhaseAction, BuyAction, \
    SellAction, DiscoverChoiceAction, HeroPowerAction, FreezeDecision, HeroDiscoverAction, SummonAction, Action, \
    PlaySpellAction
from hearthstone.simulator.core.card_pool import PrintingPress
from hearthstone.simulator.core.cards import CardLocation
from hearthstone.simulator.core.hero import EmptyHero
from hearthstone.simulator.core.hero_pool import VALHALLA
from hearthstone.simulator.core.monster_types import MONSTER_TYPES
from hearthstone.simulator.core.player import Player, StoreIndex, HandIndex, BoardIndex, DiscoverIndex, SpellIndex
from hearthstone.simulator.core.spell_pool import ALL_SPELLS
from hearthstone.training.pytorch.encoding.state_encoding import State, LocatedCard, Feature, ScalarFeature, \
    OnehotFeature, CombinedFeature, ListOfFeatures, SortedByValueFeature, EncodedActionSet, Encoder, InvalidAction, \
    ActionSet, SummonComponent, ActionComponent, SpellComponent

MAX_ENCODED_STORE = 7
MAX_ENCODED_HAND = 10
MAX_ENCODED_SPELLS = 10
MAX_ENCODED_BOARD = 7
MAX_ENCODED_DISCOVER = 3


def enum_to_int(value: Optional[enum.Enum]) -> int:
    if value is not None:
        return value.value
    else:
        return 0


CARD_TYPE_TO_INT = {card_type: ind for ind, card_type in enumerate(PrintingPress.cards)}
INT_TO_CARD_TYPE = {ind: card_type for ind, card_type in enumerate(PrintingPress.cards)}
HERO_TYPE_TO_INT = {hero_type: ind for ind, hero_type in enumerate([EmptyHero] + VALHALLA)}
INT_TO_HERO_TYPE = {ind: hero_type for ind, hero_type in enumerate([EmptyHero] + VALHALLA)}
SPELL_TYPE_TO_INT = {spell_type: ind for ind, spell_type in enumerate(ALL_SPELLS)}
INT_TO_SPELL_TYPE = {ind: spell_type for ind, spell_type in enumerate(ALL_SPELLS)}


def default_card_encoding() -> Feature:
    """
    Default encoder for type `LocatedCard`.
    """
    return CombinedFeature([
        ScalarFeature(lambda card: 1.0),  # Present
        ScalarFeature(lambda card: float(card.card.tier)),
        ScalarFeature(lambda card: float(card.card.attack)),
        ScalarFeature(lambda card: float(card.card.health)),
        ScalarFeature(lambda card: float(card.card.golden)),
        ScalarFeature(lambda card: float(card.card.taunt)),
        ScalarFeature(lambda card: float(card.card.divine_shield)),
        ScalarFeature(lambda card: float(card.card.poisonous)),
        ScalarFeature(lambda card: float(card.card.magnetic)),
        ScalarFeature(lambda card: float(card.card.windfury)),
        ScalarFeature(lambda card: float(card.card.reborn)),
        ScalarFeature(lambda card: float(bool(card.card.deathrattles))),
        ScalarFeature(lambda card: float(bool(card.card.battlecry))),
        OnehotFeature(lambda card: CARD_TYPE_TO_INT[type(card.card)], len(PrintingPress.cards) + 1),
        OnehotFeature(lambda card: enum_to_int(card.card.monster_type), len(MONSTER_TYPES) + 1),
        OnehotFeature(lambda card: enum_to_int(card.location), len(CardLocation) + 1),
    ])


def default_spell_encoding() -> Feature:
    """
    Default encoder for type `Spell`.
    """
    return CombinedFeature([
        ScalarFeature(lambda spell: 1.0),  # Present
        ScalarFeature(lambda spell: float(spell.cost)),
        ScalarFeature(lambda spell: float(spell.tier or 0)),
        OnehotFeature(lambda spell: SPELL_TYPE_TO_INT[type(spell)], len(ALL_SPELLS) + 1),
    ])


def default_player_encoding() -> Feature:
    """
    Default encoder for the player level features (non-card features).

    Encodes a `Player`.
    """
    return CombinedFeature([
        ScalarFeature(lambda player: float(player.tavern.turn_count)),
        ScalarFeature(lambda player: float(player.health)),
        ScalarFeature(lambda player: float(player.coins)),
        ScalarFeature(lambda player: float(player.tavern_tier)),
        ScalarFeature(lambda player: float(len(player.in_play))),
        ScalarFeature(lambda player: float(len(player.hand))),
        ScalarFeature(lambda player: float(len(player.store))),
        ScalarFeature(lambda player: float(len(player.spells))),
        OnehotFeature(lambda player: HERO_TYPE_TO_INT[type(player.hero)], len(VALHALLA) + 1),
        ScalarFeature(lambda player: float(player.tavern.get_paired_opponent(player).health)),
        ScalarFeature(lambda player: float(player.tavern.get_paired_opponent(player).tavern_tier)),
        SortedByValueFeature(lambda player: [p.tavern_tier for name, p in player.tavern.players.items()], 8),
        SortedByValueFeature(lambda player: [p.health for name, p in player.tavern.players.items()], 8),
    ])


def default_spells_encoding() -> Feature:
    """
    Default encoder for the spell-level features.

    Encodes a `Player`.
    """
    return ListOfFeatures(lambda player: player.spells, default_spell_encoding(), MAX_ENCODED_SPELLS)


def default_cards_encoding() -> Feature:
    """
    Default encoder for the card-level features.

    Encodes a `Player`.
    """
    return CombinedFeature([
        ListOfFeatures(
            lambda player: [LocatedCard(card, CardLocation.STORE) for card in player.store],
            default_card_encoding(), MAX_ENCODED_STORE),
        ListOfFeatures(
            lambda player: [LocatedCard(card, CardLocation.HAND) for card in player.hand],
            default_card_encoding(), MAX_ENCODED_HAND),
        ListOfFeatures(
            lambda player: [LocatedCard(card, CardLocation.BOARD) for card in player.in_play],
            default_card_encoding(), MAX_ENCODED_BOARD),
        ListOfFeatures(
            lambda player: [LocatedCard(card, CardLocation.DISCOVER) for card in
                            (player.discover_queue[0] if player.discover_queue else [])],
            default_card_encoding(), MAX_ENCODED_DISCOVER)
    ])


DEFAULT_PLAYER_ENCODING = default_player_encoding()
DEFAULT_CARDS_ENCODING = default_cards_encoding()
DEFAULT_SPELLS_ENCODING = default_spells_encoding()


def store_indices() -> List[StoreIndex]:
    return [StoreIndex(i) for i in range(MAX_ENCODED_STORE)]


def hand_indices() -> List[HandIndex]:
    return [HandIndex(i) for i in range(MAX_ENCODED_HAND)]


def spell_indices() -> List[SpellIndex]:
    return [SpellIndex(i) for i in range(MAX_ENCODED_SPELLS)]


def board_indices() -> List[BoardIndex]:
    return [BoardIndex(i) for i in range(MAX_ENCODED_BOARD)]


def discover_indices() -> List[DiscoverIndex]:
    return [DiscoverIndex(i) for i in range(MAX_ENCODED_DISCOVER)]


def _all_actions() -> ActionSet:
    player_action_set = [TavernUpgradeAction(), RerollAction(), HeroPowerAction(),
                         HeroDiscoverAction(DiscoverIndex(0)), EndPhaseAction(FreezeDecision.NO_FREEZE),
                         EndPhaseAction(FreezeDecision.FREEZE),
                         EndPhaseAction(FreezeDecision.UNFREEZE)]
    store_action_set = [
        [BuyAction(index), InvalidAction(), InvalidAction(), InvalidAction(), HeroPowerAction(store_target=index)]
        for index in store_indices()]
    hand_action_set = [
        [InvalidAction(), SummonComponent(index), InvalidAction(), InvalidAction(),
         InvalidAction()] for index in hand_indices()]
    board_action_set = [
        [InvalidAction(), InvalidAction(), InvalidAction(), SellAction(index), HeroPowerAction(board_target=index)]
        for index in board_indices()]
    discover_action_set = [
        [InvalidAction(), InvalidAction(),
         InvalidAction(), DiscoverChoiceAction(index), InvalidAction()] for index in
        discover_indices()]

    battlecry_no_target_action_set = [SummonAction(hand_index, []) for hand_index in hand_indices()]
    battlecry_action_set = [[SummonAction(hand_index, [board_index]) for board_index in board_indices()]
                            for hand_index in hand_indices()]

    spell_action_set = [SpellComponent(index) for index in spell_indices()]
    no_target_spell_action_set = [PlaySpellAction(index) for index in spell_indices()]
    store_target_spell_action_set = [
        [PlaySpellAction(index, store_target=store_index) for store_index in store_indices()] for index in
        spell_indices()]
    board_target_spell_action_set = [
        [PlaySpellAction(index, board_target=board_index) for board_index in board_indices()] for index in
        spell_indices()]

    return ActionSet(player_action_set,
                     store_action_set + hand_action_set + board_action_set + discover_action_set,
                     battlecry_no_target_action_set, battlecry_action_set, spell_action_set, no_target_spell_action_set,
                     store_target_spell_action_set, board_target_spell_action_set)


ALL_ACTIONS = _all_actions()


def _all_actions_list():
    result = []
    for player_action in ALL_ACTIONS.player_action_set:
        result.append(player_action)
    for card_actions in ALL_ACTIONS.card_action_set:
        for card_action in card_actions:
            result.append(card_action)
    for card_action in ALL_ACTIONS.spell_action_set:
        result.append(card_action)
    return result

INVERTED_ACTIONS_LIST = _all_actions_list()

ALL_ACTIONS_DICT: Dict[str, int] = {str(action): index for index, action in enumerate(INVERTED_ACTIONS_LIST)}

class DefaultEncoder(Encoder):
    def encode_state(self, player: Player) -> State:
        player_tensor = torch.from_numpy(DEFAULT_PLAYER_ENCODING.encode(player))
        cards_tensor = torch.from_numpy(DEFAULT_CARDS_ENCODING.encode(player))
        spells_tensor = torch.from_numpy(DEFAULT_SPELLS_ENCODING.encode(player))
        return State(player_tensor, cards_tensor, spells_tensor)

    def encode_valid_actions(self, player: Player, rearrange_phase: bool = False) -> EncodedActionSet:
        actions = ALL_ACTIONS

        player_action_tensor = torch.tensor([action.valid(player) for action in actions.player_action_set])
        cards_action_array = np.ndarray((len(actions.card_action_set), len(actions.card_action_set[0])), dtype=bool)
        for i, card_actions in enumerate(actions.card_action_set):
            for j, action in enumerate(card_actions):
                cards_action_array[i, j] = action.valid(player)
        cards_action_tensor = torch.from_numpy(cards_action_array)
        no_target_battlecry_tensor = torch.tensor(
            [action.valid(player) for action in actions.battlecry_no_target_action_set])
        battlecry_target_array = np.zeros((len(actions.battlecry_action_set), len(actions.battlecry_action_set[0])), dtype=bool)
        for i, card_actions in enumerate(actions.battlecry_action_set):
            for j, action in enumerate(card_actions):
                battlecry_target_array[i, j] = action.valid(player)
        battlecry_target_tensor = torch.from_numpy(battlecry_target_array)
        spell_action_tensor = torch.tensor([action.valid(player) for action in actions.spell_action_set]).unsqueeze(-1)
        no_target_spell_action_tensor = torch.tensor([action.valid(player) for action in actions.no_target_spell_action_set])
        store_target_spells_action_array = np.ndarray((len(actions.spell_store_action_set), len(actions.spell_store_action_set[0])), dtype=bool)
        for i, spell_actions in enumerate(actions.spell_store_action_set):
            for j, action in enumerate(spell_actions):
                store_target_spells_action_array[i, j] = action.valid(player)
        store_target_spells_action_tensor = torch.from_numpy(store_target_spells_action_array)
        board_target_spells_action_array = np.ndarray(
            (len(actions.spell_board_action_set), len(actions.spell_board_action_set[0])), dtype=bool)
        for i, spell_actions in enumerate(actions.spell_board_action_set):
            for j, action in enumerate(spell_actions):
                board_target_spells_action_array[i, j] = action.valid(player)
        board_target_spells_action_tensor = torch.from_numpy(board_target_spells_action_array)

        cards_to_rearrange = torch.tensor(
            len(player.in_play), dtype=torch.long)

        return EncodedActionSet(player_action_tensor, cards_action_tensor, no_target_battlecry_tensor,
                                battlecry_target_tensor, spell_action_tensor, no_target_spell_action_tensor,
                                store_target_spells_action_tensor, board_target_spells_action_tensor,
                                torch.tensor(rearrange_phase),
                                cards_to_rearrange, 0, MAX_ENCODED_STORE, MAX_ENCODED_STORE + MAX_ENCODED_HAND)

    def player_encoding(self) -> Feature:
        return DEFAULT_PLAYER_ENCODING

    def cards_encoding(self) -> Feature:
        return DEFAULT_CARDS_ENCODING

    def spells_encoding(self) -> Feature:
        return DEFAULT_SPELLS_ENCODING

    def action_encoding_size(self) -> int:
        player_action_size = len(ALL_ACTIONS.player_action_set)
        card_action_size = len(ALL_ACTIONS.card_action_set) * len(ALL_ACTIONS.card_action_set[0])
        spells_action_size = len(ALL_ACTIONS.spell_action_set)
        return player_action_size + card_action_size + spells_action_size

    def get_action_component_index(self, action: Union[ActionComponent, Action]) -> int:
        if isinstance(action, SummonAction):
            action = SummonComponent(action.index)
        if isinstance(action, PlaySpellAction):
            action = SpellComponent(action.index)
        return ALL_ACTIONS_DICT[str(action)]

    def get_indexed_action_component(self, index: int) -> ActionComponent:
        return INVERTED_ACTIONS_LIST[index]
