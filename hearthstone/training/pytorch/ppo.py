import logging
import os
import random
import time
from datetime import datetime
from typing import List, Dict, Union, NewType

import torch
import trueskill
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from hearthstone.ladder.ladder import Contestant, load_ratings, ContestantAgentGenerator
from hearthstone.simulator.agent import RearrangeCardsAction, BuyAction, EndPhaseAction, SellAction, SummonAction, \
    RerollAction, DiscoverChoiceAction, TavernUpgradeAction, TripleRewardsAction
from hearthstone.simulator.core.hero import EmptyHero
from hearthstone.simulator.core.tavern import Tavern
from hearthstone.training.pytorch.agents.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.encoding import shared_tensor_pool_encoder
from hearthstone.training.pytorch.encoding.default_encoder import \
    DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING, EncodedActionSet, \
    DefaultEncoder
from hearthstone.training.pytorch.encoding.shared_tensor_pool_encoder import SharedTensorPoolEncoder
from hearthstone.training.pytorch.encoding.state_encoding import State
from hearthstone.training.pytorch.gae import GAEAnnotator
from hearthstone.training.pytorch.networks import save_load
from hearthstone.training.pytorch.networks.save_load import create_net, load_from_saved
from hearthstone.training.pytorch.policy_gradient import tensorize_batch, easy_contestants, easiest_contestants, \
    easier_contestants
from hearthstone.training.pytorch.replay import ActorCriticGameStepInfo
from hearthstone.training.pytorch.replay_buffer import EpochBuffer
from hearthstone.training.pytorch.surveillance import GlobalStepContext
from hearthstone.training.pytorch.worker.worker import Worker
from hearthstone.training.pytorch.worker.worker_pool import WorkerPool

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

        self.expensive_tensorboard = False
        self.histogram_tensorboard = False
        # Total number of gradient descent steps we've taken. (for reporting to tensorboard)
        self.global_step = 0
        # Number of games we have plotted
        self.games_plotted = 0

        # global_step at time of last model export.
        self.last_exported_step = 0
        self.export_path = "../../../data/learning/pytorch/saved_models/{}".format(self.hparams['export.path'])

        # Encoder is shared to reuse tensors passed between processes
        self.encoder = DefaultEncoder()
        if self.hparams['parallelism.method'] in ("process", "batch"):
            self.encoder = SharedTensorPoolEncoder(self.encoder, self.hparams['parallelism.method']=="process")

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
            stop_early = self.learn_minibatch(epoch, minibatch_idx, tensorboard, optimizer, learning_net, minibatch, policy_weight, entropy_weight,
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
        transition_batch = tensorize_batch(minibatch, self.get_device())

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

        value_loss = torch.max(value_error.pow(2), clipped_value_error.pow(2)).mean()

        entropy_loss = (entropy_weight * action_log_probs).mean()
        main_dist_kl_divergence = (debug.component_policy.exp() * (
                    debug.component_policy - transition_batch.debug_component_policy)).sum(dim=1).mean()
        approx_kl_divergence = -log_ratio.mean()
        if self.histogram_tensorboard:
            tensorboard.add_histogram("policy", torch.exp(action_log_probs), self.global_step)
            masked_reward = transition_batch.reward.masked_select(transition_batch.is_terminal)
            if masked_reward.size()[0]:
                tensorboard.add_histogram("reward",
                                          transition_batch.reward.masked_select(transition_batch.is_terminal),
                                          self.global_step)
            tensorboard.add_histogram("value/current", value, self.global_step)
            tensorboard.add_histogram("value/next", transition_batch.value, self.global_step)
            tensorboard.add_histogram("advantage/unclipped", advantage, self.global_step)
            tensorboard.add_histogram("advantage/normalized", normalized_advantage, self.global_step)
            tensorboard.add_histogram("policy_loss/unclipped", unclipped_policy_loss, self.global_step)
            tensorboard.add_histogram("policy_loss/clipped", clipped_policy_loss, self.global_step)
            tensorboard.add_histogram("policy_ratio/unclipped", ratio, self.global_step)
            tensorboard.add_histogram("policy_ratio/clipped", clipped_ratio, self.global_step)
        tensorboard.add_scalar("avg_reward",
                               transition_batch.reward.masked_select(transition_batch.is_terminal).float().mean(),
                               self.global_step)
        tensorboard.add_scalar("avg_value", value.mean(), self.global_step)
        tensorboard.add_scalar("avg_advantage/unnormalized", advantage.mean(), self.global_step)
        tensorboard.add_scalar("avg_advantage/normalized", normalized_advantage.mean(), self.global_step)
        tensorboard.add_scalar("avg_value_error", value_error.mean(), self.global_step)
        tensorboard.add_scalar("loss/policy", policy_loss, self.global_step)
        tensorboard.add_scalar("loss/policy/main_dist",
                               torch.max(clipped_policy_loss, unclipped_policy_loss).masked_select(
                                   transition_batch.valid_actions.rearrange_phase.logical_not()).mean(),
                               self.global_step)
        tensorboard.add_scalar("loss/policy/rearrange",
                               torch.max(clipped_policy_loss, unclipped_policy_loss).masked_select(
                                   transition_batch.valid_actions.rearrange_phase).mean(), self.global_step)
        tensorboard.add_scalar("loss/value", value_loss, self.global_step)
        tensorboard.add_scalar("loss/entropy", entropy_loss, self.global_step)
        tensorboard.add_scalar("loss/entropy/main_dist", entropy_weight * action_log_probs.masked_select(
            transition_batch.valid_actions.rearrange_phase.logical_not()).mean(), self.global_step)
        tensorboard.add_scalar("loss/entropy/rearrange", entropy_weight * action_log_probs.masked_select(
            transition_batch.valid_actions.rearrange_phase).mean(), self.global_step)
        tensorboard.add_scalar("kl_divergence/main_dist", main_dist_kl_divergence, self.global_step)
        tensorboard.add_scalar("kl_divergence/approx", approx_kl_divergence, self.global_step)
        tensorboard.add_scalar("kl_divergence/approx/main_dist", -log_ratio.masked_select(
            transition_batch.valid_actions.rearrange_phase.logical_not()).mean(), self.global_step)
        tensorboard.add_scalar("kl_divergence/approx/rearrange", -log_ratio.masked_select(
            transition_batch.valid_actions.rearrange_phase).mean(), self.global_step)

        if epoch == 0 and minibatch_idx == 0:
            tensorboard.add_scalar("kl_divergence/before_learning_main_dist", main_dist_kl_divergence, self.global_step)
            tensorboard.add_scalar("kl_divergence/before_learning_approx", approx_kl_divergence, self.global_step)
        if approx_kl_divergence > self.hparams['approx_kl_limit']:
            tensorboard.add_scalar("kl_divergence/early_stopped_epoch", epoch, self.global_step)
            tensorboard.add_scalar("kl_divergence/early_stopped_main_dist", main_dist_kl_divergence, self.global_step)
            tensorboard.add_scalar("kl_divergence/early_stopped_approx", approx_kl_divergence, self.global_step)
        tensorboard.add_scalar("avg_policy_loss/unclipped", unclipped_policy_loss.mean(), self.global_step)
        tensorboard.add_scalar("avg_policy_loss/clipped", clipped_policy_loss.mean(), self.global_step)
        tensorboard.add_scalar("actions/endphase_terminal", transition_batch.is_terminal.sum(), self.global_step)
        tensorboard.add_scalar("actions/endphase", sum(type(action) is EndPhaseAction for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/endphase_freeze", sum(type(action) is EndPhaseAction and action.freeze for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/rearrange", sum(type(action) is RearrangeCardsAction for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/buy", sum(type(action) is BuyAction for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/sell", sum(type(action) is SellAction for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/summon", sum(type(action) is SummonAction for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/reroll", sum(type(action) is RerollAction for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/upgrade", sum(type(action) is TavernUpgradeAction for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/triple_rewards", sum(type(action) is TripleRewardsAction for action in transition_batch.action), self.global_step)
        tensorboard.add_scalar("actions/discover", sum(type(action) is DiscoverChoiceAction for action in transition_batch.action), self.global_step)

        mean_0_return = transition_batch.retn - transition_batch.retn.mean()
        mean_0_value = value - value.mean()
        mean_0_diff = transition_batch.retn - value
        mean_0_diff -= mean_0_diff.mean()
        tensorboard.add_scalar("critic_explanation/explained_variance", (1 - mean_0_diff.pow(2).mean()) /
                               mean_0_return.pow(2).mean(), self.global_step)
        tensorboard.add_scalar("critic_explanation/correlation", (mean_0_return * mean_0_value).mean() / (
                mean_0_return.pow(2).mean() * mean_0_value.pow(2).mean()).sqrt(), self.global_step)

        loss = policy_loss * policy_weight + value_loss + entropy_loss
        optimizer.zero_grad()
        loss.backward()
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(learning_net.parameters(), gradient_clipping)
        optimizer.step()
        if self.expensive_tensorboard:
            for tag, parm in learning_net.named_parameters():
                tensorboard.add_histogram(f"gradients_{tag}/train", parm.grad.data, self.global_step)
        return approx_kl_divergence > self.hparams['approx_kl_limit']

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
        batch_size = self.hparams['batch_size']

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
            optimizer = optim.Adam(learning_net.parameters(), lr=self.hparams["adam_lr"])
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

        for _ in range(1000000):
            learning_net.eval()
            if self.hparams['parallelism.method']:
                worker_pool.play_games(learning_bot_contestant, other_contestants, self.hparams['game_size'])
            else:
                Worker(learning_bot_contestant, other_contestants, self.hparams['game_size'], replay_buffer,
                       gae_annotator, tensorboard, self).play_game()
            if len(replay_buffer) >= batch_size:
                print(len(replay_buffer))
                learning_net.train()
                print(f"Running {self.hparams['ppo_epochs']} epochs of PPO optimization...")
                for i in range(self.hparams["ppo_epochs"]):
                    self.handle_export(learning_bot_contestant, learning_net, other_contestants)
                    stop_early = self.learn_epoch(i, tensorboard, optimizer, learning_net, replay_buffer,
                                     self.hparams['minibatch_size'],
                                     self.hparams["policy_weight"],
                                     self.hparams["entropy_weight"], self.hparams["ppo_epsilon"],
                                     self.hparams["gradient_clipping"],
                                     self.hparams["normalize_advantage"])
                    if stop_early:
                        break
                if self.hparams['parallelism.method'] == "process":
                    replay_buffer.recycle(shared_tensor_pool_encoder.global_process_tensor_queue)
                elif self.hparams['parallelism.method'] == "batch":
                    # replay_buffer.recycle(shared_tensor_pool_encoder.global_thread_tensor_queue)
                    replay_buffer.clear()
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


def main():
    ppo_learner = PPOLearner(PPOHyperparameters({
        "resume": False,
        'resume.from': '2021-01-06T01:18:59.151166',
        'export.enabled': True,
        'export.period_epochs': 200,
        'export.path': datetime.now().isoformat(),
        'opponents.initial': 'easiest',
        'opponents.self_play.enabled': True,
        'opponents.self_play.only_champions': True,
        'opponents.self_play.remove_weakest': False,
        'opponents.max_pool_size': 7,
        'adam_lr': 0.0001,
        'batch_size': 512,
        'minibatch_size': 512,
        'cuda': True,
        'entropy_weight': 0.001,
        'gae_gamma': 0.999,
        'gae_lambda': 0.9,
        'game_size': 8,
        'gradient_clipping': 0.5,
        'approx_kl_limit': 0.015,
        'nn.architecture': 'transformer',
        'nn.state_encoder': 'Default',
        'nn.hidden_layers': 3,
        'nn.hidden_size': 128,
        'nn.activation': 'gelu',
        'nn.shared': False,
        'nn.encoding.redundant': True,
        'nn.encoding.normalize': True,
        'nn.encoding.normalize.gamma': 0.999,
        'normalize_advantage': True,
        'parallelism.num_workers': 64,
        'parallelism.method': "batch",
        'optimizer': 'adam',
        'policy_weight': 0.581166675499831,
        'ppo_epochs': 8,
        'ppo_epsilon': 0.2}))

    ppo_learner.run()


if __name__ == '__main__':
    main()
