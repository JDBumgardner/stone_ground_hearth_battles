import asyncio
import logging
import random
from typing import Optional, Any, Dict

import torch
from torch import nn

from hearthstone.simulator.agent.actions import StandardAction, DiscoverChoiceAction, RearrangeCardsAction, \
    HeroDiscoverAction, Action
from hearthstone.simulator.agent.agent import AnnotatingAgent
from hearthstone.training.pytorch.encoding.default_encoder import \
    EncodedActionSet
from hearthstone.training.pytorch.encoding.state_encoding import State, Encoder
from hearthstone.training.pytorch.replay import ActorCriticGameStepInfo

logger = logging.getLogger(__name__)


class PytorchBot(AnnotatingAgent):
    def __init__(self, net: nn.Module, encoder: Encoder, annotate: bool = True, device: Optional[torch.device] = None):
        self.authors = ["Jeremy Salwen"]
        self.net: nn.Module = net
        self.encoder: Encoder = encoder
        self.annotate = annotate
        self.device = device

    async def async_net(self, *args, **kwargs):
        result = self.net(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        else:
            return result

    async def act(self, player: 'Player', rearrange_cards: bool) -> (Action, ActorCriticGameStepInfo):
        with torch.no_grad():
            encoded_state: State = self.encoder.encode_state(player).to(self.device)
            valid_actions_mask: EncodedActionSet = self.encoder.encode_valid_actions(player, rearrange_cards).to(
                self.device)
            actions, action_log_probs, value, debug = await self.async_net(
                State(encoded_state.player_tensor.unsqueeze(0),
                      encoded_state.cards_tensor.unsqueeze(0)),
                EncodedActionSet(
                    valid_actions_mask.player_action_tensor.unsqueeze(0),
                    valid_actions_mask.card_action_tensor.unsqueeze(0),
                    valid_actions_mask.rearrange_phase.unsqueeze(0),
                    valid_actions_mask.cards_to_rearrange.unsqueeze(0)),
                None)
            assert (len(actions) == 1)
            action = actions[0]
            ac_game_step_info = None
            if self.annotate:
                ac_game_step_info = ActorCriticGameStepInfo(
                    state=encoded_state,
                    valid_actions=valid_actions_mask,
                    action=action,
                    action_log_prob=float(action_log_probs[0]),
                    value=float(value),
                    gae_info=None,
                    debug=debug
                )

            return action, ac_game_step_info

    async def annotated_buy_phase_action(self, player: 'Player') -> (StandardAction, ActorCriticGameStepInfo):
        action, ac_game_step_info = await self.act(player, False)
        assert isinstance(action, StandardAction)
        return action, ac_game_step_info

    async def annotated_rearrange_cards(self, player: 'Player') -> (RearrangeCardsAction, ActorCriticGameStepInfo):
        action, ac_game_step_info = await self.act(player, True)
        assert isinstance(action, RearrangeCardsAction)
        return action, ac_game_step_info

    async def annotated_discover_choice_action(self, player: 'Player') -> (
    DiscoverChoiceAction, ActorCriticGameStepInfo):
        action, ac_game_step_info = await self.act(player, False)
        assert isinstance(action, DiscoverChoiceAction), action
        return action, ac_game_step_info

    async def annotated_hero_discover_action(self, player: 'Player') -> ('HeroDiscoverAction', ActorCriticGameStepInfo):
        return HeroDiscoverAction(random.choice(range(len(player.hero.discover_choices)))), None

    async def game_over(self, player: 'Player', ranking: int) -> Dict[str, Any]:
        return {'ranking': ranking}
