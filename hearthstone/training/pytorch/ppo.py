import logging
import os
import random
import time
from datetime import datetime
from math import sqrt
from typing import List, Dict, Union, NewType

import torch
import trueskill
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from hearthstone.ladder.ladder import Contestant, load_ratings, ContestantAgentGenerator
from hearthstone.simulator.agent.actions import RearrangeCardsAction, BuyAction, EndPhaseAction, SellAction, \
    SummonAction, \
    RerollAction, DiscoverChoiceAction, TavernUpgradeAction, TripleRewardsAction, HeroPowerAction, FreezeDecision, \
    BananaAction, RedeemGoldCoinAction
from hearthstone.simulator.core.hero import EmptyHero
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.training.pytorch.agents.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.encoding import shared_tensor_pool_encoder
from hearthstone.training.pytorch.encoding.default_encoder import \
    EncodedActionSet, \
    DefaultEncoder
from hearthstone.training.pytorch.encoding.shared_tensor_pool_encoder import SharedTensorPoolEncoder
from hearthstone.training.pytorch.encoding.state_encoding import State
from hearthstone.training.pytorch.gae import GAEAnnotator
from hearthstone.training.pytorch.networks import save_load
from hearthstone.training.pytorch.networks.running_norm import WelfordAggregator
from hearthstone.training.pytorch.networks.save_load import create_net, load_from_saved
from hearthstone.training.pytorch.policy_gradient import tensorize_batch, easy_contestants, easiest_contestants, \
    easier_contestants, TransitionBatch
from hearthstone.training.pytorch.replay import ActorCriticGameStepInfo, ActorCriticGameStepDebugInfo
from hearthstone.training.pytorch.replay_buffer import EpochBuffer
from hearthstone.training.pytorch.surveillance import GlobalStepContext
from hearthstone.training.pytorch.worker.distributed.worker_pool import DistributedWorkerPool
from hearthstone.training.pytorch.worker.postprocessing import ExperiencePostProcessor
from hearthstone.training.pytorch.worker.single_machine.worker import Worker
from hearthstone.training.pytorch.worker.single_machine.worker_pool import WorkerPool

PPOHyperparameters = NewType('PPOHyperparameters', Dict[str, Union[str, int, float]])


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

        # Total number of gradient descent steps we've taken. (for reporting to tensorboard)
        self.global_step = 0
        # Number of games we have plotted
        self.games_plotted = 0

        # global_step at time of last model export.
        self.last_exported_step = 0
        self.export_path = "../../../data/learning/pytorch/saved_models/{}".format(self.hparams['export.path'])

        # Encoder is shared to reuse tensors passed between processes
        self.encoder = DefaultEncoder()
        if self.hparams['parallelism.shared_tensor_pool']:
            self.encoder = SharedTensorPoolEncoder(self.encoder, self.hparams['parallelism.method'] == "process")

        self.tensorboard_accum = PPOTensorboard(False, False)

    def get_global_step(self) -> int:
        return self.global_step

    def should_plot(self) -> bool:
        do_plot = self.games_plotted % 10 == 0
        self.games_plotted += 1
        return do_plot

    def learn_epoch(self,
                    epoch: int,
                    tensorboard: SummaryWriter,
                    optimizer: optim.Optimizer,
                    learning_net: nn.Module,
                    replay_buffer: EpochBuffer,
                    minibatch_size: int,
                    policy_weight: float,
                    entropy_weight: float,
                    ppo_epsilon: float,
                    gradient_clipping: float,
                    normalize_advantage: bool
                    ) -> bool:
        for minibatch_idx, minibatch in enumerate(replay_buffer.sample_minibatches(minibatch_size)):
            stop_early = self.learn_minibatch(epoch, minibatch_idx, tensorboard, optimizer, learning_net, minibatch,
                                              policy_weight, entropy_weight,
                                              ppo_epsilon,
                                              gradient_clipping, normalize_advantage)
            self.global_step += 1
            if stop_early:
                return True
        return False

    def learn_minibatch(self,
                        epoch: int,
                        minibatch_idx: int,
                        tensorboard: SummaryWriter,
                        optimizer: optim.Optimizer,
                        learning_net: nn.Module,
                        minibatch: List[ActorCriticGameStepInfo],
                        policy_weight: float,
                        entropy_weight: float,
                        ppo_epsilon: float,
                        gradient_clipping: float,
                        normalize_advantage: bool) -> bool:
        """
        This does one step of gradient descent on a mini batch of transitions.

        Args:
            tensorboard: Visualization framework
            optimizer: PyTorch optimizer, e.g. Adam.
            learning_net: The NN. A wise man once said, the learning net catch the smart fish.
            minibatch: A minibatch of ActorCriticGameStepInfo to learn from.
            policy_weight: The weight of the policy relative to the value
            entropy_weight: The weight of entropy loss, which is used for regularization.
            ppo_epsilon: Clips the difference between the current and next value between +/- ppo_epsilon.
            gradient_clipping: Clip the norm of the gradient if it's above this threshold.
            normalize_advantage: Whether to batch normal the advantage with a mean of 0 and stdev of 1.
        :returns Bool, whether to stop early due to kl constraint being exceeded
        """
        self.tensorboard_accum.reset()
        approx_kl_divergence_welford = WelfordAggregator(torch.Size())
        optimizer.zero_grad()
        # We have to split into sub batches, since the minibatch might not fit onto our wimpy GPU's memory.
        for i in range(0, len(minibatch), self.hparams['batch.max_in_memory']):
            sub_batch = minibatch[i: i + self.hparams['batch.max_in_memory']]
            transition_batch = tensorize_batch(sub_batch, self.get_device())

            actions, action_log_probs, value, debug = learning_net(transition_batch.state,
                                                                   transition_batch.valid_actions,
                                                                   transition_batch.action)

            # The advantage is the difference between the expected value before taking the action and the value after updating
            advantage = transition_batch.gae_return - transition_batch.value

            log_ratio = (action_log_probs - transition_batch.action_log_prob)
            ratio = torch.exp(log_ratio)
            clipped_ratio = ratio.clamp(1 - ppo_epsilon, 1 + ppo_epsilon)

            normalized_advantage: torch.Tensor = advantage
            if normalize_advantage:
                normalized_advantage = (normalized_advantage - normalized_advantage.mean()) / (
                        normalized_advantage.std(unbiased=False) + 1e-7)
            clipped_policy_loss = - clipped_ratio * normalized_advantage
            unclipped_policy_loss = - ratio * normalized_advantage
            policy_loss = torch.max(clipped_policy_loss, unclipped_policy_loss).mean()

            value_error = transition_batch.retn - value
            # Clip the value error to be within ppo_epsilon of the advantage at the time that the action was taken.
            clipped_value_error = transition_batch.retn - transition_batch.value + torch.clamp(
                transition_batch.value - value,
                -ppo_epsilon, ppo_epsilon)

            if self.hparams['ppo_clip_value']:
                value_loss = torch.max(value_error.pow(2), clipped_value_error.pow(2)).mean()
            else:
                value_loss = value_error.pow(2).mean()

            entropy_loss = (entropy_weight * action_log_probs).mean()
            approx_kl_divergence_welford.update(-log_ratio.mean())

            loss = policy_loss * policy_weight + value_loss + entropy_loss
            loss.backward()
            self.tensorboard_accum.update(learning_net, transition_batch, debug, value,
                                          advantage, normalized_advantage, value_error,
                                          policy_loss, unclipped_policy_loss, clipped_policy_loss,
                                          value_loss, entropy_loss, entropy_weight,
                                          action_log_probs, log_ratio)

        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(learning_net.parameters(), gradient_clipping)
        optimizer.step()
        early_stopped = approx_kl_divergence_welford.mean() > self.hparams['approx_kl_limit']
        self.tensorboard_accum.flush(tensorboard, epoch, minibatch_idx, approx_kl_divergence_welford.mean(),
                                     early_stopped,
                                     self.hparams['batch.max_in_memory'], len(minibatch), optimizer.state_dict(),
                                     self.global_step)
        return early_stopped

    def get_device(self) -> torch.device:
        if self.hparams['cuda'] and torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')

    def add_graph_to_tensorboard(self, tensorboard: SummaryWriter, learning_net: nn.Module):
        tavern = Tavern()
        tavern.add_player_with_hero("Dummy", EmptyHero())
        tavern.add_player_with_hero("Other", EmptyHero())
        tavern.buying_step()
        player = list(tavern.players.values())[0]

        encoder = DefaultEncoder()
        encoded_state: State = encoder.encode_state(player).to(self.get_device())
        valid_actions_mask: EncodedActionSet = encoder.encode_valid_actions(player).to(self.get_device())

        tensorboard.add_graph(learning_net, input_to_model=(State(encoded_state.player_tensor.unsqueeze(0),
                                                                  encoded_state.cards_tensor.unsqueeze(0)),
                                                            EncodedActionSet(
                                                                valid_actions_mask.player_action_tensor.unsqueeze(0),
                                                                valid_actions_mask.card_action_tensor.unsqueeze(0))))

    def load_latest_saved_versions(self, run, n) -> Dict[int, nn.Module]:
        resume_from_dir = "../../../data/learning/pytorch/saved_models/{}".format(run)
        models = os.listdir(resume_from_dir)
        top_n_models = sorted([int(model) for model in models], reverse=True)[:n]
        return {model: load_from_saved("{}/{}".format(resume_from_dir, model), self.hparams)
                for model in top_n_models}

    def get_initial_contestants(self) -> List[Contestant]:
        if self.hparams['resume']:
            return [Contestant("LoadedBot_{}".format(model),
                               ContestantAgentGenerator(PytorchBot,
                                                        net=net,
                                                        encoder=self.encoder,
                                                        annotate=False,
                                                        device=self.get_device(),
                                                        )) for model, net in
                    self.load_latest_saved_versions(self.hparams['resume.from'],
                                                    self.hparams['opponents.max_pool_size']).items()]
        if self.hparams['opponents.initial'] == "easiest":
            return easiest_contestants()
        if self.hparams['opponents.initial'] == "easier":
            return easier_contestants()
        if self.hparams['opponents.initial'] == "easy":
            return easy_contestants()
        assert False

    def handle_export(self, learning_bot_contestant: Contestant, learning_net, other_contestants):
        if self.global_step > self.last_exported_step + self.hparams['export.period_epochs']:
            if self.hparams['export.enabled']:
                self.last_exported_step = self.global_step
                state_dict = learning_net.state_dict()
                torch.save(state_dict, "{}/{}".format(self.export_path, str(self.global_step)))
                if self.hparams['opponents.self_play.enabled']:
                    # If it's +-sigma confidence interval puts it better than the next best bot, we make a new
                    # copy.
                    if learning_bot_contestant.trueskill.mu - learning_bot_contestant.trueskill.sigma > max(
                            c.trueskill.mu for c in other_contestants) or not \
                            self.hparams['opponents.self_play.only_champions']:
                        frozen_clone = save_load.create_net(self.hparams)
                        frozen_clone.load_state_dict(state_dict)
                        frozen_clone.eval()
                        while len(other_contestants) + 1 > self.hparams['opponents.max_pool_size']:
                            if self.hparams['opponents.self_play.remove_weakest']:
                                min_trueskill = min(c.trueskill.mu for c in other_contestants)
                                weakest_opponents = [opp for opp in other_contestants if
                                                     opp.trueskill.mu == min_trueskill]
                                other_contestants.remove(weakest_opponents[0])
                            else:
                                other_contestants.pop(random.randrange(0, len(other_contestants)))
                        other_contestants.append(Contestant(
                            "{}_{}".format(learning_bot_contestant.name, self.global_step),
                            ContestantAgentGenerator(PytorchBot,
                                                     net=frozen_clone,
                                                     encoder=self.encoder,
                                                     annotate=False,
                                                     device=self.get_device())
                        ))

    def run(self):
        start_time = time.time()
        last_reported_time = start_time
        min_replay_buffer_size = self.hparams['batch.min_replay_buffer_size']

        device = self.get_device()
        tensorboard = SummaryWriter(f"../../../data/learning/pytorch/tensorboard/{self.hparams['export.path']}")
        logging.getLogger().setLevel(logging.INFO)

        if self.hparams['export.enabled']:
            os.mkdir(self.export_path)

        if self.hparams['resume']:
            self.global_step, learning_net = \
                list(self.load_latest_saved_versions(self.hparams['resume.from'], 1).items())[0]
        else:
            learning_net = create_net(self.hparams)
        # This is broken on CUDA and we are too lazy to debug.
        # self.add_graph_to_tensorboard(tensorboard, learning_net)

        # Set gradient descent algorithm
        if self.hparams["optimizer"] == "adam":
            # {https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam}
            optimizer = optim.Adam(learning_net.parameters(), lr=self.hparams["adam.lr"],
                                   weight_decay=self.hparams["adam.weight_decay"])
        elif self.hparams["optimizer"] == "sgd":
            # {https://en.wikipedia.org/wiki/Stochastic_gradient_descent}
            optimizer = optim.SGD(learning_net.parameters(), lr=self.hparams["sgd_lr"],
                                  momentum=self.hparams["sgd_momentum"],
                                  nesterov=True)
        else:
            assert False

        learning_bot_name = "LearningBot"
        replay_buffer = EpochBuffer(learning_bot_name)
        learning_bot_contestant = Contestant(
            learning_bot_name,
            ContestantAgentGenerator(PytorchBot,
                                     net=learning_net,
                                     encoder=self.encoder,
                                     annotate=True,
                                     device=device)
        )
        # Rating starts a 14, which is how the randomly initialized pytorch bot performs.
        learning_bot_contestant.trueskill = trueskill.Rating(14)
        # Reuse standings from the current leaderboard.

        other_contestants = self.get_initial_contestants()
        load_ratings(other_contestants, "../../../data/standings/8p.json")
        gae_annotator = GAEAnnotator(learning_bot_name, self.hparams['gae_gamma'], self.hparams['gae_lambda'])
        if self.hparams['parallelism.method']:
            if self.hparams['parallelism.method'] == "distributed":
                worker_pool = DistributedWorkerPool(num_workers=self.hparams['parallelism.num_workers'],
                                                    games_per_worker=self.hparams[
                                                        'parallelism.distributed.games_per_worker'],
                                                    use_batched_inference=True,
                                                    max_batch_size=self.hparams['batch.max_in_memory'],
                                                    replay_sink=ExperiencePostProcessor(replay_buffer, gae_annotator,
                                                                                        tensorboard, self),
                                                    device=self.get_device()
                                                    )
            else:
                worker_pool = WorkerPool(self.hparams['parallelism.num_workers'],
                                         replay_buffer,
                                         gae_annotator,
                                         self.encoder,
                                         tensorboard,
                                         self,
                                         self.hparams['parallelism.method'] == "process",
                                         self.hparams['parallelism.method'] == "batch",
                                         self.get_device(),
                                         )

        for i in range(1000000):
            learning_net.eval()
            if self.hparams['parallelism.method']:
                worker_pool.play_games(learning_bot_contestant, other_contestants, self.hparams['game_size'])
            else:
                Worker(learning_bot_contestant, other_contestants, self.hparams['game_size'], replay_buffer,
                       gae_annotator, tensorboard, self).play_game()
            # Warmup games for normalization layer are discarded.
            if self.hparams['resume'] and self.hparams['nn.encoding.normalize'] and i == 0:
                replay_buffer.clear()
                continue
            if len(replay_buffer) >= min_replay_buffer_size:
                print(len(replay_buffer))
                learning_net.train()
                print(f"Running {self.hparams['ppo_epochs']} epochs of PPO optimization...")
                for i in range(self.hparams["ppo_epochs"]):
                    self.handle_export(learning_bot_contestant, learning_net, other_contestants)
                    stop_early = self.learn_epoch(i, tensorboard, optimizer, learning_net, replay_buffer,
                                                  self.hparams['batch.minibatch_size'],
                                                  self.hparams["policy_weight"],
                                                  self.hparams["entropy_weight"], self.hparams["ppo_epsilon"],
                                                  self.hparams["gradient_clipping"],
                                                  self.hparams["normalize_advantage"])
                    if stop_early:
                        break
                if self.hparams['parallelism.shared_tensor_pool']:
                    if self.hparams['parallelism.method'] == "process":
                        replay_buffer.recycle(shared_tensor_pool_encoder.global_process_tensor_queue)
                    elif self.hparams['parallelism.method'] in ("thread", "batch"):
                        replay_buffer.recycle(shared_tensor_pool_encoder.global_thread_tensor_queue)
                    else:
                        assert False, "Unidentified parallelism method."
                else:
                    replay_buffer.clear()

            time_elapsed = int(time.time() - start_time)
            tensorboard.add_scalar("rating/elo", learning_bot_contestant.elo, global_step=self.global_step)
            tensorboard.add_scalar("rating/trueskill_mu", learning_bot_contestant.trueskill.mu,
                                   global_step=self.global_step)
            tensorboard.add_scalar("rating/trueskill_sigma", learning_bot_contestant.trueskill.sigma,
                                   global_step=self.global_step)
            if self.early_stopper:
                if time.time() - last_reported_time > 5:
                    last_reported_time = time.time()
                    self.early_stopper.report(learning_bot_contestant.trueskill.mu, time_elapsed)
                if self.early_stopper.should_prune():
                    break
            if self.time_limit_secs and time_elapsed > self.time_limit_secs:
                break

        tensorboard.add_hparams(hparam_dict=self.hparams,
                                metric_dict={"optuna_trueskill": learning_bot_contestant.trueskill.mu})
        tensorboard.close()
        return learning_bot_contestant.trueskill.mu


class PPOTensorboard:
    def __init__(self, expensive_metrics: bool, histogram_metrics: bool):
        self.expensive_metrics = expensive_metrics
        self.histogram_metrics = histogram_metrics
        self.reset()

        # These are kept across multiple gradient steps.
        self.gradient_signal_mag = WelfordAggregator(torch.Size())
        self.gradient_noise_mag = WelfordAggregator(torch.Size())

    def reset(self):
        self.count = 0
        self.reward_welford = WelfordAggregator(torch.Size())
        self.value_welford = WelfordAggregator(torch.Size())
        self.advantage_welford = WelfordAggregator(torch.Size())
        self.normalized_advantage_welford = WelfordAggregator(torch.Size())
        self.value_error_welford = WelfordAggregator(torch.Size())
        self.policy_loss_sum = 0
        self.policy_loss_unclipped_sum = 0
        self.policy_loss_clipped_sum = 0
        self.policy_loss_main_dist_welford = WelfordAggregator(torch.Size())
        self.policy_loss_rearrange_welford = WelfordAggregator(torch.Size())
        self.value_loss_sum = 0
        self.entropy_loss_sum = 0
        self.entropy_loss_main_dist_approx_welford = WelfordAggregator(torch.Size())
        self.entropy_loss_main_dist_exact_welford = WelfordAggregator(torch.Size())
        self.entropy_loss_main_dist_min_welford = WelfordAggregator(torch.Size())
        self.entropy_loss_rearrange_approx_welford = WelfordAggregator(torch.Size())
        self.entropy_loss_rearrange_min_welford = WelfordAggregator(torch.Size())
        self.kl_divergence_main_dist_exact_welford = WelfordAggregator(torch.Size())
        self.kl_divergence_main_dist_approx_welford = WelfordAggregator(torch.Size())
        self.kl_divergence_rearrange_approx_welford = WelfordAggregator(torch.Size())
        self.return_welford = WelfordAggregator(torch.Size())
        self.gae_return_welford = WelfordAggregator(torch.Size())
        self.terminal_value_welford = WelfordAggregator(torch.Size())
        self.terminal_value_error_welford = WelfordAggregator(torch.Size())
        self.actions = []
        self.terminal_action_count = 0
        self.small_batch_grad_norm = 0.0
        self.big_batch_grad_norm = 0.0

    def update(self, net: nn.Module, transition_batch: TransitionBatch, debug: ActorCriticGameStepDebugInfo,
               value: torch.Tensor,
               advantage: torch.Tensor, normalized_advantage: torch.Tensor, value_error: torch.Tensor,
               policy_loss: torch.Tensor, unclipped_policy_loss: torch.Tensor, clipped_policy_loss: torch.Tensor,
               value_loss: torch.Tensor, entropy_loss: torch.Tensor, entropy_weight: float,
               action_log_probs: torch.Tensor, log_ratio: torch.Tensor):

        self.big_batch_grad_norm = 0.0
        for name, param in net.named_parameters():
            if param.grad is not None:
                self.big_batch_grad_norm += float(param.grad.pow(2).sum())
        if self.count == 0:
            self.small_batch_grad_norm = self.big_batch_grad_norm

        new_count = transition_batch.value.shape[0]
        self.count += new_count
        self.reward_welford.update(transition_batch.reward.masked_select(transition_batch.is_terminal).float())
        self.value_welford.update(value)
        self.advantage_welford.update(advantage)
        self.normalized_advantage_welford.update(normalized_advantage)
        self.value_error_welford.update(value_error)
        self.policy_loss_sum += policy_loss * new_count
        self.policy_loss_unclipped_sum += unclipped_policy_loss.sum()
        self.policy_loss_clipped_sum += clipped_policy_loss.sum()
        self.policy_loss_main_dist_welford.update(torch.max(clipped_policy_loss, unclipped_policy_loss).masked_select(
            transition_batch.valid_actions.rearrange_phase.logical_not()))
        self.policy_loss_rearrange_welford.update(torch.max(clipped_policy_loss, unclipped_policy_loss).masked_select(
            transition_batch.valid_actions.rearrange_phase))
        self.value_loss_sum += value_loss * new_count
        self.entropy_loss_sum += entropy_loss * new_count
        self.entropy_loss_main_dist_approx_welford.update(entropy_weight * action_log_probs.masked_select(
            transition_batch.valid_actions.rearrange_phase.logical_not()))
        self.entropy_loss_main_dist_exact_welford.update(
            entropy_weight * (debug.component_policy.exp() * debug.component_policy).sum(
                dim=1).masked_select(
                transition_batch.valid_actions.rearrange_phase.logical_not()))
        self.entropy_loss_main_dist_min_welford.update(
            - entropy_weight * (transition_batch.valid_actions.player_action_tensor.sum(
                dim=1) + transition_batch.valid_actions.card_action_tensor.flatten(1).sum(
                dim=1)).float().log().masked_select(
                transition_batch.valid_actions.rearrange_phase.logical_not()))
        self.entropy_loss_rearrange_approx_welford.update(entropy_weight * action_log_probs.masked_select(
            transition_batch.valid_actions.rearrange_phase))
        self.entropy_loss_rearrange_min_welford.update(- entropy_weight *
                                                       (transition_batch.valid_actions.cards_to_rearrange[:,
                                                        1] + 1).masked_select(
                                                           transition_batch.valid_actions.rearrange_phase).float().lgamma())
        self.kl_divergence_main_dist_exact_welford.update((debug.component_policy.exp() * (
                debug.component_policy - transition_batch.debug_component_policy)).sum(dim=1).masked_select(
            transition_batch.valid_actions.rearrange_phase.logical_not()))
        self.kl_divergence_main_dist_approx_welford.update(-log_ratio.masked_select(
            transition_batch.valid_actions.rearrange_phase.logical_not()))
        self.kl_divergence_rearrange_approx_welford.update(-log_ratio.masked_select(
            transition_batch.valid_actions.rearrange_phase))
        self.return_welford.update(transition_batch.retn)
        self.gae_return_welford.update(transition_batch.gae_return)
        self.terminal_value_welford.update(value.masked_select(transition_batch.is_terminal))
        self.terminal_value_error_welford.update(
            value.masked_select(transition_batch.is_terminal) - transition_batch.reward.masked_select(
                transition_batch.is_terminal).float())

        self.actions += transition_batch.action
        self.terminal_action_count += transition_batch.is_terminal.sum()

    def flush(self, tensorboard: SummaryWriter, epoch: int, minibatch_idx: int, approx_kl_divergence: torch.Tensor,
              early_stopped: bool,
              small_batch_size: int, big_batch_size: int, optimizer_state, step):
        tensorboard.add_scalar("reward/mean", self.reward_welford.mean(), step)
        tensorboard.add_scalar("reward/stddev", self.reward_welford.stdev(), step)
        tensorboard.add_scalar("value/mean", self.value_welford.mean(), step)
        tensorboard.add_scalar("value/stddev", self.value_welford.stdev(), step)
        tensorboard.add_scalar("advantage/mean/unnormalized", self.advantage_welford.mean(), step)
        tensorboard.add_scalar("advantage/mean/normalized", self.normalized_advantage_welford.mean(), step)
        tensorboard.add_scalar("advantage/stddev/unnormalized", self.advantage_welford.stdev(), step)
        tensorboard.add_scalar("advantage/stddev/normalized", self.normalized_advantage_welford.stdev(), step)
        tensorboard.add_scalar("value_error/mean", self.value_error_welford.mean(), step)
        tensorboard.add_scalar("value_error/stddev", self.value_error_welford.stdev(), step)
        tensorboard.add_scalar("loss/policy", self.policy_loss_sum / self.count, step)
        tensorboard.add_scalar("loss/policy/main_dist", self.policy_loss_main_dist_welford.mean(), step)
        tensorboard.add_scalar("loss/policy/rearrange", self.policy_loss_rearrange_welford.mean(), step)
        tensorboard.add_scalar("loss/value", self.value_loss_sum / self.count, step)
        tensorboard.add_scalar("loss/entropy", self.entropy_loss_sum / self.count, step)
        tensorboard.add_scalars("loss/entropy/main_dist",
                                {"approx": self.entropy_loss_main_dist_approx_welford.mean(),
                                 "exact": self.entropy_loss_main_dist_exact_welford.mean(),
                                 "min": self.entropy_loss_main_dist_min_welford.mean(),
                                 }, step)
        tensorboard.add_scalars("loss/entropy/rearrange",
                                {"approx": self.entropy_loss_rearrange_approx_welford.mean(),
                                 "min": self.entropy_loss_rearrange_min_welford.mean(),
                                 }, step)
        tensorboard.add_scalars("kl_divergence/main_dist",
                                {
                                    "exact": self.kl_divergence_main_dist_exact_welford.mean(),
                                    "approx": self.kl_divergence_main_dist_approx_welford.mean(),
                                }, step)

        tensorboard.add_scalar("kl_divergence/approx", approx_kl_divergence, step)
        tensorboard.add_scalar("kl_divergence/approx/rearrange", self.kl_divergence_rearrange_approx_welford.mean(),
                               step)

        if epoch == 0 and minibatch_idx == 0:
            tensorboard.add_scalar("kl_divergence/before_learning_main_dist",
                                   self.kl_divergence_main_dist_exact_welford.mean(), step)
            tensorboard.add_scalar("kl_divergence/before_learning_approx", approx_kl_divergence, step)
        if early_stopped:
            tensorboard.add_scalar("kl_divergence/early_stopped_epoch", epoch, step)
            tensorboard.add_scalar("kl_divergence/early_stopped_main_dist",
                                   self.kl_divergence_main_dist_exact_welford.mean(), step)
            tensorboard.add_scalar("kl_divergence/early_stopped_approx", approx_kl_divergence, step)
        tensorboard.add_scalar("avg_policy_loss/unclipped", self.policy_loss_unclipped_sum / self.count, step)
        tensorboard.add_scalar("avg_policy_loss/clipped", self.policy_loss_clipped_sum / self.count, step)

        tensorboard.add_scalar("actions/endphase_terminal", self.terminal_action_count, step)
        tensorboard.add_scalar("actions/endphase",
                               sum(type(action) is EndPhaseAction for action in self.actions), step)
        tensorboard.add_scalar("actions/endphase_no_freeze", sum(
            type(action) is EndPhaseAction and action.freeze == FreezeDecision.NO_FREEZE for action in self.actions),
                               step)
        tensorboard.add_scalar("actions/endphase_freeze", sum(
            type(action) is EndPhaseAction and action.freeze == FreezeDecision.FREEZE for action in self.actions), step)
        tensorboard.add_scalar("actions/endphase_unfreeze", sum(
            type(action) is EndPhaseAction and action.freeze == FreezeDecision.UNFREEZE for action in self.actions),
                               step)
        tensorboard.add_scalar("actions/rearrange",
                               sum(type(action) is RearrangeCardsAction for action in self.actions), step)
        tensorboard.add_scalar("actions/buy", sum(type(action) is BuyAction for action in self.actions), step)
        tensorboard.add_scalar("actions/sell", sum(type(action) is SellAction for action in self.actions), step)
        tensorboard.add_scalar("actions/summon",
                               sum(type(action) is SummonAction for action in self.actions), step)
        tensorboard.add_scalar("actions/reroll",
                               sum(type(action) is RerollAction for action in self.actions), step)
        tensorboard.add_scalar("actions/upgrade",
                               sum(type(action) is TavernUpgradeAction for action in self.actions), step)
        tensorboard.add_scalar("actions/triple_rewards",
                               sum(type(action) is TripleRewardsAction for action in self.actions), step)
        tensorboard.add_scalar("actions/discover",
                               sum(type(action) is DiscoverChoiceAction for action in self.actions), step)
        tensorboard.add_scalar("actions/hero_power",
                               sum(type(action) is HeroPowerAction for action in self.actions), step)
        tensorboard.add_scalar("actions/banana",
                               sum(type(action) is BananaAction for action in self.actions), step)
        tensorboard.add_scalar("actions/redeem_gold_coin",
                               sum(type(action) is RedeemGoldCoinAction for action in self.actions), step)

        tensorboard.add_scalar("critic_explanation/correlation", (
                self.value_welford.variance() + self.return_welford.variance() - self.value_error_welford.variance()) / (
                                       2 * self.value_welford.stdev() * self.return_welford.stdev()), step)

        tensorboard.add_scalar("critic_explanation/residual_return_variance",
                               self.value_error_welford.variance() / self.return_welford.variance(), step)
        tensorboard.add_scalar("critic_explanation/residual_gae_variance",
                               self.advantage_welford.variance() / self.gae_return_welford.variance(), step)

        tensorboard.add_scalar("critic_explanation/terminal_correlation",
                               (
                                       self.terminal_value_welford.variance() + self.reward_welford.variance() - self.terminal_value_error_welford.variance()) / (
                                       2 * self.terminal_value_welford.stdev() * self.reward_welford.stdev()
                               ), step)

        tensorboard.add_scalar("gradients/small_batch_l2", sqrt(self.small_batch_grad_norm) / small_batch_size, step)
        if small_batch_size != big_batch_size:
            tensorboard.add_scalar("gradients/big_batch_l2", sqrt(self.big_batch_grad_norm) / big_batch_size, step)
            gradient_signal_estimate = (
                                               self.big_batch_grad_norm / big_batch_size - self.small_batch_grad_norm / small_batch_size) / (
                                               big_batch_size - small_batch_size)
            gradient_noise_estimate = (
                                              self.small_batch_grad_norm / small_batch_size ** 2 - self.big_batch_grad_norm / big_batch_size ** 2) / (
                                              1.0 / small_batch_size - 1.0 / self.big_batch_grad_norm)
            tensorboard.add_scalar("gradients/gradient_signal_est", gradient_signal_estimate, step)
            tensorboard.add_scalar("gradients/gradient_noise_est", gradient_noise_estimate, step)
            self.gradient_signal_mag.update(torch.tensor(gradient_signal_estimate))
            self.gradient_noise_mag.update(torch.tensor(gradient_noise_estimate))
            self.gradient_signal_mag.decay(1 - 0.00000002 * big_batch_size)
            self.gradient_signal_mag.decay(1 - 0.00000002 * big_batch_size)
            tensorboard.add_scalar("gradients/gradient_signal_est/rolling", self.gradient_signal_mag.mean(), step)
            tensorboard.add_scalar("gradients/gradient_noise_est/rolling", self.gradient_noise_mag.mean(), step)
            tensorboard.add_scalar("gradients/noise_scale",
                                   self.gradient_noise_mag.mean() / self.gradient_signal_mag.mean(), step)

        s = optimizer_state['state']
        v0 = torch.cat([s['exp_avg_sq'].reshape(-1) for _, s in s.items()]).norm()
        m0 = torch.cat([s['exp_avg'].reshape(-1) for _, s in s.items()]).norm()
        tensorboard.add_scalar("gradients/adam_gradient_noise_scale", big_batch_size * (v0 - m0 ** 2).item(), step)

        if self.histogram_metrics:
            tensorboard.add_histogram("policy", torch.exp(action_log_probs), step)
            masked_reward = transition_batch.reward.masked_select(transition_batch.is_terminal)
            if masked_reward.size()[0]:
                tensorboard.add_histogram("reward",
                                          transition_batch.reward.masked_select(transition_batch.is_terminal),
                                          step)
            tensorboard.add_histogram("value/current", value, step)
            tensorboard.add_histogram("value/next", transition_batch.value, step)
            tensorboard.add_histogram("advantage/unclipped", advantage, step)
            tensorboard.add_histogram("advantage/normalized", normalized_advantage, step)
            tensorboard.add_histogram("policy_loss/unclipped", unclipped_policy_loss, step)
            tensorboard.add_histogram("policy_loss/clipped", clipped_policy_loss, step)
            tensorboard.add_histogram("policy_ratio/unclipped", ratio, step)
            tensorboard.add_histogram("policy_ratio/clipped", clipped_ratio, step)

        if self.expensive_metrics:
            for tag, parm in learning_net.named_parameters():
                tensorboard.add_histogram(f"gradients_{tag}/train", parm.grad.data, step)


def main():
    ppo_learner = PPOLearner(PPOHyperparameters({
        "resume": False,
        'resume.from': '2021-02-05T17:56:25.030000',
        'export.enabled': True,
        'export.period_epochs': 200,
        'export.path': datetime.now().isoformat(),
        'opponents.initial': 'easiest',
        'opponents.self_play.enabled': True,
        'opponents.self_play.only_champions': True,
        'opponents.self_play.remove_weakest': True,
        'opponents.max_pool_size': 7,
        'adam.lr': 1e-4,
        'adam.weight_decay': 5e-5,
        'batch.min_replay_buffer_size': 400000,
        'batch.minibatch_size': 16384,
        'batch.max_in_memory': 1024,
        'cuda': True,
        'entropy_weight': 0.001,
        'gae_gamma': 0.999,
        'gae_lambda': 0.95,
        'game_size': 8,
        'gradient_clipping': 0.5,
        'approx_kl_limit': 0.03,
        'nn.architecture': 'transformer',
        'nn.state_encoder': 'Default',
        'nn.hidden_layers': 3,
        'nn.hidden_size': 128,
        'nn.activation': 'gelu',
        'nn.shared': False,
        'nn.encoding.redundant': True,
        'nn.encoding.normalize': False,
        'nn.encoding.normalize.gamma': 0.999999,
        'normalize_advantage': True,
        'parallelism.num_workers': 6,
        'parallelism.method': 'distributed',
        'parallelism.shared_tensor_pool': False,
        'parallelism.distributed.games_per_worker': 128,
        'optimizer': 'adam',
        'policy_weight': 0.5,
        'ppo_epochs': 8,
        'ppo_epsilon': 0.2,
        'ppo_clip_value': False}))

    ppo_learner.run()


if __name__ == '__main__':
    main()
