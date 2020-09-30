import collections
from typing import Optional, List

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from hearthstone.agent import Action
from hearthstone.training.pytorch.hearthstone_state_encoder import State, EncodedActionSet, get_indexed_action, \
    encode_player, encode_valid_actions, Transition, get_action_index
from hearthstone.training.pytorch.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.replay_buffer import ReplayBuffer, logger


class Parasite:
    def on_hero_choice_action(self, player: 'Player', hero: 'Hero'):
        pass

    def on_rearrange_cards(self, player: 'Player', new_board: List['MonsterCard']):
        pass

    def on_buy_phase_action(self, player: 'Player', action: Action, policy: torch.Tensor, value: torch.Tensor):
        pass

    def on_discover_choice_action(self, player: 'Player', card: 'MonsterCard'):
        pass

    def on_game_over(self, player: 'Player', ranking: int):
        pass


class SurveiledPytorchBot(PytorchBot):
    def __init__(self, net: nn.Module, parasites: List[Parasite] = None):
        """
        A pytorch bot that has attached listeners who get notified every time the bot takes an action
        :param net: The Pytorch value/policy module for this bot.
        :param parasites: A list of parasites who get notified whenever this bot takes an action.
        """
        super().__init__(net)
        self.parasites = parasites or []

    async def buy_phase_action(self, player: 'Player') -> Action:
        policy, value = self.policy_and_value(player)
        probs = torch.exp(policy[0])
        action_index = Categorical(probs).sample()
        action = get_indexed_action(int(action_index))
        if not action.valid(player):
            logger.debug("No! Bad Citizen!")
            logger.debug("This IDE is lit")
        else:
            for parasite in self.parasites:
                parasite.on_buy_phase_action(player, action, policy, value)
        return action

    async def game_over(self, player: 'Player', ranking: int):
        for parasite in self.parasites:
            parasite.on_game_over(player, ranking)


class ReplayBufferSaver(Parasite):
    def __init__(self, replay_buffer: ReplayBuffer):
        """
        Puts transitions into the replay buffer.

        Args:
            replay_buffer: Buffer of transitions.
        """
        self.replay_buffer = replay_buffer
        self.last_state: Optional[State] = None
        self.last_action: Optional[int] = None
        self.last_action_prob: Optional[float] = None
        self.last_valid_actions: Optional[EncodedActionSet] = None
        self.last_value: Optional[float] = None

    def on_buy_phase_action(self, player: 'Player', action: Action, policy: torch.Tensor, value: torch.Tensor):
        action_index = get_action_index(action)
        if not action.valid(player):
            logger.debug("No! Bad Citizen!")
            logger.debug("This IDE is lit")
        else:
            new_state = encode_player(player)
            if self.last_state is not None:
                self.remember_result(new_state, 0, False)
            self.last_state = encode_player(player)
            self.last_valid_actions = encode_valid_actions(player)
            self.last_action = int(action_index)
            self.last_action_prob = float(policy[0][action_index])
            self.last_value = float(value)
        return action

    def on_game_over(self, player: 'Player', ranking: int):
        if self.last_state is not None:
            self.remember_result(encode_player(player), 3.5 - ranking, True)

    def remember_result(self, new_state, reward, is_terminal):
        self.replay_buffer.push(Transition(self.last_state, self.last_valid_actions,
                                           self.last_action, self.last_action_prob,
                                           self.last_value,
                                           new_state, reward, is_terminal))


class GlobalStepContext:
    def get_global_step(self) -> int:
        raise NotImplemented("Not Implemented")

    def should_plot(self) -> bool:
        raise NotImplemented("Note Implemented")


class TensorboardGamePlotter(Parasite):
    def __init__(self, tensorboard: SummaryWriter, global_step_context: GlobalStepContext):
        self.tensorboard = tensorboard
        self.healths = []
        self.dead_players = []
        self.avg_enemy_healths = []
        self.values = []
        self.global_step_context = global_step_context
        self.action_types = []

    def on_buy_phase_action(self, player: 'Player', action: Action, policy: torch.Tensor, value: torch.Tensor):
        self.update_gamestate(player, value)
        self.action_types.append(type(action).__name__)

    def update_gamestate(self, player: 'Player', value):
        self.healths.append(player.health)
        self.avg_enemy_healths.append((sum(max(p.health, 0) for name, p in player.tavern.players.items()) - player.health) / 7.0)
        self.dead_players.append(len(player.tavern.losers)-3.5)
        if value is not None:
            self.values.append(float(value))

    def on_game_over(self, player: 'Player', ranking: int):
        self.update_gamestate(player, None)

        figure, ax1 = plt.subplots()
        lns1 = ax1.plot(self.dead_players, label="Dead Players", color="tab:green")
        lns2 = ax1.plot(self.values, label="Value", color="black")
        lns3 = ax1.plot(len(self.values), 3.5 - ranking, label="Reward", marker="*", color="black", linestyle="none")
        ax1.grid(True, axis="x")
        plt.legend()
        ax2 = ax1.twinx()
        ax2.set_ylim([-5, 45])
        lns4 = ax2.plot(self.healths, label="Health", color='tab:blue')
        lns5 = ax2.plot(self.avg_enemy_healths, label="Avg Enemy Health", color='tab:purple')

        leg = lns1 + lns2 + lns3 + lns4 + lns5
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc=0)

        self.tensorboard.add_figure("game_state", figure, global_step=self.global_step_context.get_global_step())
        plt.close()
        figure, ax = plt.subplots()

        c = collections.Counter(self.action_types)
        c = sorted(c.items())
        plt.bar([i[0] for i in c], [i[1] for i in c])
        plt.title("Action Probablity By Type")
        plt.xlabel("Action Type")
        plt.ylabel("Frequency")
        ax.set_xticks(range(len(c)))
        ax.set_xticklabels([i[0] for i in c])
        figure.autofmt_xdate()
        self.tensorboard.add_figure("action_type_distribution", figure, global_step=self.global_step_context.get_global_step())
        plt.close()
        self.values.clear()
