import logging
import random
import time
from datetime import datetime
from typing import List, Dict, Union, NewType

import torch
import trueskill
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from hearthstone.simulator.host import RoundRobinHost
from hearthstone.ladder.ladder import Contestant, update_ratings, load_ratings, print_standings
from hearthstone.training.pytorch.hearthstone_state_encoder import Transition, get_indexed_action, \
    DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING
from hearthstone.training.pytorch.networks.feedforward_net import HearthstoneFFNet
from hearthstone.training.pytorch.networks.transformer_net import HearthstoneTransformerNet
from hearthstone.training.pytorch.policy_gradient import tensorize_batch, easiest_contestants
from hearthstone.training.pytorch.replay_buffer import ReplayBuffer, NormalizingReplayBuffer
from hearthstone.training.pytorch.surveillance import SurveiledPytorchBot, ReplayBufferSaver, GlobalStepContext
from hearthstone.training.pytorch.tensorboard_altair import TensorboardAltairPlotter

PPOHyperparameters = NewType('PPOHyperparameters', Dict[str, Union[str, int, float]])


class Worker:
    def __init__(self, learning_bot_contestant: Contestant, other_contestants: List[Contestant], replay_buffer):
        """
        Worker is responsible for setting up games where the learning bot plays against a random set of opponents and
        provides a way to step through the games one action at a time.

        Args:
            learning_bot_contestant (Contestant):
            other_contestants (List[Contestant]):
        """
        self.other_contestants = other_contestants
        self.learning_bot_contestant = learning_bot_contestant
        self.host = None
        self.round_contestants = None
        self.one_round_generator = None
        self.learning_bot_agent = None
        self.replay_buffer = replay_buffer
        self._start_new_game()

    def _start_new_game(self):
        self.round_contestants = [self.learning_bot_contestant] + random.sample(self.other_contestants, k=7)
        self.host = RoundRobinHost(
            {contestant.name: contestant.agent_generator() for contestant in self.round_contestants})
        self.learning_bot_agent = self.host.agents[self.learning_bot_contestant.name]
        self.host.start_game()
        self.one_round_generator = self.host.play_round_generator()

    def play_step(self):
        last_replay_buffer_position = self.replay_buffer.position
        while self.replay_buffer.position == last_replay_buffer_position:
            try:
                next(self.one_round_generator)
            except StopIteration as e:
                if self.host.game_over():
                    winner_names = list(reversed([name for name, player in self.host.tavern.losers]))
                    print("---------------------------------------------------------------")
                    print(winner_names)
                    print(self.host.tavern.players[self.learning_bot_contestant.name].in_play)
                    ranked_contestants = sorted(self.round_contestants, key=lambda c: winner_names.index(c.name))
                    update_ratings(ranked_contestants)
                    print_standings([self.learning_bot_contestant] + self.other_contestants)
                    for contestant in self.round_contestants:
                        contestant.games_played += 1

                    self._start_new_game()
                else:
                    self.one_round_generator = self.host.play_round_generator()


class PPOLearner(GlobalStepContext):
    """
    Trains a bot using proximal policy optimization, reporting metrics to tensorboard as well as optuna trial.

    Like optimizing a tuna. Also, optional tuning.

    Args:
        hparams: the hyperparameters {link to optuner doc}
        time_limit_secs: timeout
        early_stopper: optional optuna trial that we report the progress to.

    Returns: Trueskill rating for the bot
    """

    def __init__(self, hparams: PPOHyperparameters, time_limit_secs=None, early_stopper=None):
        self.hparams = hparams
        self.time_limit_secs = time_limit_secs
        self.early_stopper = early_stopper

        self.expensive_tensorboard = False
        self.histogram_tensorboard = False
        # Total number of gradient descent steps we've taken. (for reporting to tensorboard)
        self.global_step = 0
        # Number of games we have plotted
        self.games_plotted = 0

    def get_global_step(self) -> int:
        return self.global_step

    def should_plot(self) -> bool:
        do_plot = self.games_plotted % 10 == 0
        self.games_plotted += 1
        return do_plot

    def learn(self, tensorboard: SummaryWriter, optimizer: optim.Optimizer, learning_net: nn.Module,
              replay_buffer: ReplayBuffer,
              batch_size: int, policy_weight: float, entropy_weight: float, ppo_epsilon: float,
              gradient_clipping: float,
              normalize_advantage: bool):
        """
        This does one step of gradient descent on a mini batch of transitions.

        Args:
            tensorboard: Visualization framework
            optimizer: PyTorch optimizer, e.g. Adam.
            learning_net: The NN. A wise man once said, the learning net catch the smart fish.
            replay_buffer: Buffer of observed transitions to learn from.
            batch_size: The size of your mom.
            policy_weight: The weight of the policy relative to the value
            entropy_weight: The weight of entropy loss, which is used for regularization.
            ppo_epsilon: Clips the difference between the current and next value between +/- ppo_epsilon.
            gradient_clipping: Clip the norm of the gradient if it's above this threshold.
            normalize_advantage: Whether to batch normal the advantage with a mean of 0 and stdev of 1.
        """
        transitions: List[Transition] = replay_buffer.sample(batch_size)
        transition_batch = tensorize_batch(transitions, self.get_device())
        # TODO turn off gradient here
        # Note transition_batch.valid_actions is not the set of valid actions from the next state, but we are ignoring the
        # policy network here so it doesn't matter
        with torch.no_grad():
            next_policy_, next_value = learning_net(transition_batch.next_state, transition_batch.valid_actions)

        policy, value = learning_net(transition_batch.state, transition_batch.valid_actions)
        value_target = transition_batch.reward.unsqueeze(-1) + next_value.masked_fill(
            transition_batch.is_terminal.unsqueeze(-1), 0.0)
        # The advantage is the difference between the expected value before taking the action and the value after updating
        advantage = value_target - value
        # Clip the advantage to be within ppo_epsilon of the advantage at the time that the action was taken.
        clipped_advantage = value_target - transition_batch.value + torch.clamp(transition_batch.value - value,
                                                                                -ppo_epsilon, ppo_epsilon)

        ratio = torch.exp(policy - transition_batch.action_prob.unsqueeze(-1)).gather(1,
                                                                                      transition_batch.action.unsqueeze(
                                                                                          -1))
        clipped_ratio = ratio.clamp(1 - ppo_epsilon, 1 + ppo_epsilon)

        normalized_advantage: torch.Tensor = advantage.detach()
        if normalize_advantage:
            normalized_advantage = (normalized_advantage - normalized_advantage.mean()) / (
                    normalized_advantage.std() + 1e-5)
        clipped_policy_loss = - clipped_ratio * normalized_advantage
        unclipped_policy_loss = - ratio * normalized_advantage
        policy_loss = torch.max(clipped_policy_loss, unclipped_policy_loss).mean()
        value_loss = torch.max(advantage.pow(2), clipped_advantage.pow(2)).mean()

        # Here we compute the policy only for actions which are valid.
        valid_action_tensor = torch.cat(
            (transition_batch.valid_actions.player_action_tensor.flatten(1),
             transition_batch.valid_actions.card_action_tensor.flatten(1)), dim=1)
        masked_policy = policy.masked_select(valid_action_tensor)
        entropy_loss = entropy_weight * torch.sum(masked_policy * torch.exp(masked_policy))

        if self.histogram_tensorboard:
            tensorboard.add_histogram("policy", torch.exp(masked_policy), self.global_step)
            masked_reward = transition_batch.reward.masked_select(transition_batch.is_terminal)
            if masked_reward.size()[0]:
                tensorboard.add_histogram("reward",
                                          transition_batch.reward.masked_select(transition_batch.is_terminal),
                                          self.global_step)
            tensorboard.add_histogram("value/current", value, self.global_step)
            tensorboard.add_histogram("value/next", next_value, self.global_step)
            tensorboard.add_histogram("advantage/unclipped", advantage, self.global_step)
            tensorboard.add_histogram("advantage/clipped", clipped_advantage, self.global_step)
            tensorboard.add_histogram("advantage/normalized", normalized_advantage, self.global_step)
            tensorboard.add_histogram("policy_loss/unclipped", unclipped_policy_loss, self.global_step)
            tensorboard.add_histogram("policy_loss/clipped", clipped_policy_loss, self.global_step)
            tensorboard.add_histogram("policy_ratio/unclipped", ratio, self.global_step)
            tensorboard.add_histogram("policy_ratio/clipped", clipped_ratio, self.global_step)
        tensorboard.add_text("action/train", str(get_indexed_action(int(transition_batch.action[0]))), self.global_step)
        tensorboard.add_scalar("avg_reward",
                               transition_batch.reward.masked_select(transition_batch.is_terminal).float().mean(),
                               self.global_step)
        tensorboard.add_scalar("avg_value", value.mean(), self.global_step)
        tensorboard.add_scalar("avg_advantage", advantage.mean(), self.global_step)
        tensorboard.add_scalar("policy_loss", policy_loss, self.global_step)
        tensorboard.add_scalar("value_loss", value_loss, self.global_step)
        tensorboard.add_scalar("avg_policy_loss/unclipped", unclipped_policy_loss.mean(), self.global_step)
        tensorboard.add_scalar("avg_policy_loss/clipped", clipped_policy_loss.mean(), self.global_step)

        tensorboard.add_scalar("entropy_loss", entropy_loss, self.global_step)
        loss = policy_loss * policy_weight + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(learning_net.parameters(), gradient_clipping)
        optimizer.step()
        if self.expensive_tensorboard:
            for tag, parm in learning_net.named_parameters():
                tensorboard.add_histogram(f"gradients_{tag}/train", parm.grad.data, self.global_step)

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')

    def run(self):
        start_time = time.time()
        last_reported_time = start_time
        batch_size = self.hparams['batch_size']

        device = self.get_device()
        tensorboard = SummaryWriter(f"../../../data/learning/pytorch/tensorboard/{datetime.now().isoformat()}")
        logging.getLogger().setLevel(logging.INFO)

        if self.hparams["nn_architecture"] == "feedforward":
            learning_net = HearthstoneFFNet(DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING,
                                            self.hparams["nn_hidden_layers"],
                                            self.hparams.get("nn_hidden_size") or 0,
                                            self.hparams.get("nn_shared") or False,
                                            self.hparams.get("nn_activation") or "")
        elif self.hparams["nn_architecture"] == "transformer":
            learning_net = HearthstoneTransformerNet(DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING,
                                            self.hparams["nn_hidden_layers"],
                                            self.hparams.get("nn_hidden_size") or 0,
                                            self.hparams.get("nn_shared") or False,
                                            self.hparams.get("nn_activation") or "")

        # Set gradient descent algorithm
        if self.hparams["optimizer"] == "adam":
            # {https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam}
            optimizer = optim.Adam(learning_net.parameters(), lr=self.hparams["adam_lr"])
        elif self.hparams["optimizer"] == "sgd":
            # {https://en.wikipedia.org/wiki/Stochastic_gradient_descent}
            optimizer = optim.SGD(learning_net.parameters(), lr=self.hparams["sgd_lr"],
                                  momentum=self.hparams["sgd_momentum"],
                                  nesterov=True)
        else:
            assert False


        replay_buffer_size = 10000
        if self.hparams["normalize_observations"]:
            # TODO(anyone): decide on gamma / make it a hyperparameter.
            replay_buffer = NormalizingReplayBuffer(replay_buffer_size, 0, DEFAULT_PLAYER_ENCODING,
                                                    DEFAULT_CARDS_ENCODING)
        else:
            replay_buffer = ReplayBuffer(replay_buffer_size)
        learning_bot_contestant = Contestant("LearningBot",
                                             lambda: SurveiledPytorchBot(
                                                 learning_net,
                                                 [
                                                     ReplayBufferSaver(replay_buffer),
                                                     TensorboardAltairPlotter(tensorboard, self)
                                                 ],
                                                 device)
                                             )
        # Rating starts a 14, which is how the randomly initialized pytorch bot performs.
        learning_bot_contestant.trueskill = trueskill.Rating(14)
        # Reuse standings from the current leaderboard.
        other_contestants = easiest_contestants()
        load_ratings(other_contestants, "../../../data/standings.json")

        workers = [Worker(learning_bot_contestant, other_contestants, replay_buffer) for _ in
                   range(self.hparams['num_workers'])]

        for _ in range(1000000):
            for worker in workers:
                worker.play_step()
            # print(len(replay_buffer))
            if len(replay_buffer) >= batch_size:
                for i in range(self.hparams["ppo_epochs"]):
                    self.learn(tensorboard, optimizer, learning_net, replay_buffer, batch_size,
                               self.hparams["policy_weight"],
                               self.hparams["entropy_weight"], self.hparams["ppo_epsilon"],
                               self.hparams["gradient_clipping"],
                               self.hparams["normalize_advantage"])
                    self.global_step += 1
                replay_buffer.clear()
            time_elapsed = int(time.time() - start_time)
            tensorboard.add_scalar("elo/train", learning_bot_contestant.elo, global_step=self.global_step)
            tensorboard.add_scalar("trueskill_mu/train", learning_bot_contestant.trueskill.mu, global_step=self.global_step)
            if self.early_stopper:
                if time.time() - last_reported_time > 5:
                    last_reported_time = time.time()
                    self.early_stopper.report(learning_bot_contestant.trueskill.mu, time_elapsed)
                if self.early_stopper.should_prune():
                    break
            if self.time_limit_secs and time_elapsed > time_limit_secs:
                break

        tensorboard.add_hparams(hparam_dict=self.hparams,
                                metric_dict={"optuna_trueskill": learning_bot_contestant.trueskill.mu})
        tensorboard.close()
        return learning_bot_contestant.trueskill.mu


def main():
    ppo_learner = PPOLearner(PPOHyperparameters({'adam_lr': 0.000698178899316577,
                                                 'batch_size': 269,
                                                 'entropy_weight': 3.20049705838473e-05,
                                                 'gradient_clipping': 100,
                                                 'nn_architecture': 'transformer',
                                                 'nn_hidden_layers': 3,
                                                 'nn_hidden_size': 256,
                                                 'nn_activation': 'gelu',
                                                 'nn_shared': 'false',
                                                 'normalize_advantage': True,
                                                 'normalize_observations': False,
                                                 'num_workers': 1,
                                                 'optimizer': 'sgd',
                                                 'sgd_lr': 0.00006,
                                                 'sgd_momentum': 0.1,
                                                 'policy_weight': 0.581166675499831,
                                                 'ppo_epochs': 8,
                                                 'ppo_epsilon': 100}))
    ppo_learner.run()


if __name__ == '__main__':
    main()
