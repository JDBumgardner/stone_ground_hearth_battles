import logging
from random import random
from typing import Optional, List

from hearthstone.agent import Agent, Action
from hearthstone.training.pytorch.hearthstone_state_encoder import Transition, State, encode_player, \
    encode_valid_actions, EncodedActionSet

logger = logging.getLogger(__name__)


class BigBrotherAgent(Agent):
    def __init__(self, citizen: Agent):
        self.citizen = citizen
        self.memory: List[Transition] = []
        self.last_state: Optional[State] = None
        self.last_action: Optional[Action] = None
        self.last_valid_actions: Optional[EncodedActionSet] = None

    def buy_phase_action(self, player: 'Player') -> Action:
        action = self.citizen.buy_phase_action(player)
        if not action.valid(player):
            logger.debug("No! Bad Citizen!")
        else:
            new_state = encode_player(player)
            if self.last_state is not None:
                self.remember_result(new_state, 0)
            self.last_state = encode_player(player)
            self.last_valid_actions = encode_valid_actions(player)
            self.last_action = action

        return action

    # TODO: handle learning discovery choices
    def discover_choice_action(self, player: 'Player') -> 'Card':
        return self.citizen.discover_choice_action(player)

    # TODO: Handle learning card ordering
    def rearrange_cards(self, player: 'Player') -> List['Card']:
        return self.citizen.rearrange_cards(player)

    def game_over(self, player: 'Player', ranking: int):
        if self.last_state is not None:
            self.remember_result(encode_player(player), 3.5 - ranking)

    def remember_result(self, new_state, reward):
        self.memory.append(Transition(self.last_state, self.last_valid_actions, self.last_action, new_state, reward))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory: List[Transition]= []
        self.position = 0

    def push(self, transition: Transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
