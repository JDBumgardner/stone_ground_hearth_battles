import collections
import random
import threading
from typing import Dict, List, Any, Tuple

import torch
from torch import nn

from hearthstone.simulator.agent.actions import StandardAction, RearrangeCardsAction, DiscoverChoiceAction, Action
from hearthstone.simulator.agent.agent import AnnotatingAgent
from hearthstone.training.common.state_encoding import EncodedActionSet, State, Encoder
from hearthstone.training.pytorch.replay import ActorCriticGameStepDebugInfo, ActorCriticGameStepInfo


class BatchedInferencePytorchBot(AnnotatingAgent):
    def __init__(self, queue: 'BatchedInferenceQueue', net_name: str, encoder: Encoder, annotate: bool = True,
                 device: torch.device = None):
        self.authors = ["Jeremy Salwen"]
        self.queue = queue
        self.net_name = net_name
        self.encoder: Encoder = encoder
        self.annotate = annotate
        self.device = device

    def act(self, player: 'Player', rearrange_cards: bool) -> (Action, ActorCriticGameStepInfo):
        encoded_state: State = self.encoder.encode_state(player)
        valid_actions_mask: EncodedActionSet = self.encoder.encode_valid_actions(player, rearrange_cards)
        future = self.queue.infer(self.net_name,
                                  encoded_state, valid_actions_mask)
        actions, action_log_probs, value, debug = future.get()
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
                debug=ActorCriticGameStepDebugInfo(
                    component_policy=debug.component_policy,
                    permutation_logits=debug.permutation_logits[:, :len(player.in_play)],
                )
            )
        return action, ac_game_step_info

    async def annotated_buy_phase_action(self, player: 'Player') -> (StandardAction, ActorCriticGameStepInfo):
        action, ac_game_step_info = self.act(player, False)
        assert isinstance(action, StandardAction)
        return action, ac_game_step_info

    async def annotated_rearrange_cards(self, player: 'Player') -> (RearrangeCardsAction, ActorCriticGameStepInfo):
        action, ac_game_step_info = self.act(player, True)
        assert isinstance(action, RearrangeCardsAction)
        return action, ac_game_step_info

    async def annotated_discover_choice_action(self, player: 'Player') -> (
    DiscoverChoiceAction, ActorCriticGameStepInfo):
        action, ac_game_step_info = self.act(player, False)
        assert isinstance(action, DiscoverChoiceAction)
        return action, ac_game_step_info

    async def game_over(self, player: 'Player', ranking: int) -> Dict[str, Any]:
        return {'ranking': ranking}


class InferenceFuture:
    def __init__(self):
        self.value = None
        self.done_flag = threading.Event()

    def set(self, value):
        self.value = value
        self.done_flag.set()

    def get(self):
        self.done_flag.wait()
        return self.value


class BatchedInferenceQueue:
    def __init__(self, nets: Dict[str, nn.Module], max_batch_size: int, device: torch.device):
        self.nets = nets
        for name_, net in nets.items():
            net.to(device)
        self.max_batch_size = max_batch_size
        self.device = device
        self.queued_tasks_by_net = {name: collections.deque() for name in nets.keys()}

        self.inference_example_count = 0
        self.inference_count = 0
        # These are the only variables accessed from multiple threads.
        self.communication_queue = collections.deque()
        self.communication_event = threading.Event()
        self.done_event = threading.Event()

    def infer(self, net_name: str, state: State, valid_actions: EncodedActionSet) -> InferenceFuture:
        future = InferenceFuture()
        self.communication_queue.append((net_name, future, (state, valid_actions)))
        self.communication_event.set()
        return future

    def _unload_communication_queue(self):
        while self.communication_queue:
            net_name, future, args = self.communication_queue.popleft()
            self.queued_tasks_by_net[net_name].append((future, args))

    def _tensorize_batch(self, batch: List[Tuple[State, EncodedActionSet]]) -> (State, EncodedActionSet):
        device = self.device
        player_tensor = torch.stack([b[0].player_tensor for b in batch], dim=0).detach()
        cards_tensor = torch.stack([b[0].cards_tensor for b in batch], dim=0).detach()
        spells_tensor = torch.stack([b[0].spells_tensor for b in batch], dim=0).detach()
        valid_player_actions_tensor = torch.stack(
            [b[1].player_action_tensor for b in batch], dim=0).detach()
        valid_card_actions_tensor = torch.stack(
            [b[1].card_action_tensor for b in batch], dim=0).detach()
        valid_no_target_battlecry_tensor = torch.stack(
            [b[1].no_target_battlecry_tensor for b in batch], dim=0).detach()
        valid_battlecry_target_tensor = torch.stack([b[1].battlecry_target_tensor for b in batch], dim=0).detach()
        valid_spell_action_tensor = torch.stack(
            [b[1].spell_action_tensor for b in batch], dim=0).detach()
        valid_no_target_spell_action_tensor = torch.stack(
            [b[1].no_target_spell_action_tensor for b in batch], dim=0).detach()
        valid_store_target_spell_action_tensor = torch.stack(
            [b[1].store_target_spell_action_tensor for b in batch], dim=0).detach()
        valid_board_target_spell_action_tensor = torch.stack(
            [b[1].board_target_spell_action_tensor for b in batch], dim=0).detach()
        rearrange_phase = torch.stack([b[1].rearrange_phase for b in batch], dim=0).detach()
        cards_to_rearrange = torch.stack(
            [b[1].cards_to_rearrange for b in batch], dim=0).detach()
        return (State(player_tensor=player_tensor.to(device),
                      cards_tensor=cards_tensor.to(device),
                      spells_tensor=spells_tensor.to(device)),
                EncodedActionSet(
                    player_action_tensor=valid_player_actions_tensor,
                    card_action_tensor=valid_card_actions_tensor,
                    no_target_battlecry_tensor=valid_no_target_battlecry_tensor,
                    battlecry_target_tensor=valid_battlecry_target_tensor,
                    spell_action_tensor=valid_spell_action_tensor,
                    no_target_spell_action_tensor=valid_no_target_spell_action_tensor,
                    store_target_spell_action_tensor=valid_store_target_spell_action_tensor,
                    board_target_spell_action_tensor=valid_board_target_spell_action_tensor,
                    rearrange_phase=rearrange_phase.to(device),
                    cards_to_rearrange=cards_to_rearrange.to(device),
                    store_start=batch[0][1].store_start,
                    hand_start=batch[0][1].hand_start,
                    board_start=batch[0][1].board_start,
                ).to(device))

    def _worker_thread(self):
        while True:
            self.communication_event.clear()
            self._unload_communication_queue()
            # Select the longest queue
            name, tasks = max(self.queued_tasks_by_net.items(), key=lambda kv: len(kv[1]))
            # Remove the first batch worth from the net specific queue
            length = min(len(tasks), self.max_batch_size)
            if length:
                batched_tasks = [tasks.popleft() for _ in range(length)]
                self.inference_count += 1
                self.inference_example_count += len(batched_tasks)

                # Run inference on batched tensor
                state_batch, valid_actions_batch = self._tensorize_batch([args for _, args in batched_tasks])
                output_actions, action_log_probs, value, debug_info = self.nets[name](state_batch, valid_actions_batch,
                                                                                      None)
                for i, (future, _) in enumerate(batched_tasks):
                    future.set((output_actions[i:i + 1],
                                action_log_probs[i:i + 1].detach(),
                                value[i:i + 1].detach(),
                                ActorCriticGameStepDebugInfo(
                                    component_policy=debug_info.component_policy[i:i + 1].detach(),
                                    permutation_logits=debug_info.permutation_logits[i:i + 1].detach(),
                                )
                                ))
            self.communication_event.wait(1)
            if self.done_event.is_set():
                return

    def start_worker_thread(self):
        inference_thread = threading.Thread(target=self._worker_thread)
        inference_thread.start()

    def kill_worker_thread(self):
        self.done_event.set()
