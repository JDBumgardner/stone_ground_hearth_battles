import logging
import random
from typing import Optional, Tuple, Any, Dict

import torch
from torch import nn, Tensor
from torch.distributions import Categorical

from hearthstone.simulator.agent import StandardAction, DiscoverChoiceAction, RearrangeCardsAction, \
    AnnotatingAgent, HeroDiscoverAction
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
        if self.device:
            self.net.to(device)

    def policy_and_value(self, player: 'Player') -> Tuple[Tensor, float]:
        encoded_state: State = self.encoder.encode_state(player).to(self.device)
        valid_actions_mask: EncodedActionSet = self.encoder.encode_valid_actions(player).to(self.device)

        policy, value = self.net(State(encoded_state.player_tensor.unsqueeze(0),
                                       encoded_state.cards_tensor.unsqueeze(0)),
                                 EncodedActionSet(valid_actions_mask.player_action_tensor.unsqueeze(0),
                                                  valid_actions_mask.card_action_tensor.unsqueeze(0)))
        return policy, value

    async def annotated_buy_phase_action(self, player: 'Player') -> (StandardAction, ActorCriticGameStepInfo):
        encoded_state: State = self.encoder.encode_state(player).to(self.device)
        valid_actions_mask: EncodedActionSet = self.encoder.encode_valid_actions(player).to(self.device)
        policy, value = self.net(State(encoded_state.player_tensor.unsqueeze(0),
                                       encoded_state.cards_tensor.unsqueeze(0)),
                                 EncodedActionSet(valid_actions_mask.player_action_tensor.unsqueeze(0),
                                                  valid_actions_mask.card_action_tensor.unsqueeze(0)))
        probs = torch.exp(policy[0])
        action_index = Categorical(probs).sample()
        action = self.encoder.get_indexed_action(int(action_index))

        ac_game_step_info = None
        if self.annotate:
            ac_game_step_info = ActorCriticGameStepInfo(
                state=encoded_state,
                valid_actions=valid_actions_mask,
                action=int(action_index),
                policy=policy[0].detach(),
                value=float(value),
                gae_info=None
            )

        return action, ac_game_step_info

    # TODO handle learning card and discover choice actions
    async def rearrange_cards(self, player: 'Player') -> RearrangeCardsAction:
        return RearrangeCardsAction(list(range(len(player.in_play))))

    async def discover_choice_action(self, player: 'Player') -> DiscoverChoiceAction:
        return DiscoverChoiceAction(random.choice(range(len(player.discover_queue[0]))))

    async def hero_discover_action(self, player: 'Player') -> 'HeroDiscoverAction':
        return HeroDiscoverAction(random.choice(range(len(player.hero.discover_choices))))

    async def game_over(self, player: 'Player', ranking: int) -> Dict[str, Any]:
        return {'ranking': ranking}

