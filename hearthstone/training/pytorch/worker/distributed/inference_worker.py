import collections
import threading
from time import sleep
from typing import Tuple, List, Optional, Dict

import torch
from torch import nn
from torch.distributed import rpc

from hearthstone.simulator.agent import Action
from hearthstone.training.pytorch.encoding.state_encoding import EncodedActionSet, State
from hearthstone.training.pytorch.policy_gradient import StateBatch
from hearthstone.training.pytorch.replay import ActorCriticGameStepDebugInfo


class InferenceWorker:
    def __init__(self, max_batch_size: int, num_inference_threads: int, device):
        self.id = rpc.get_worker_info().id

        self.max_batch_size = max_batch_size
        self.num_inference_threads = num_inference_threads
        self.device = device

        self.nets: Dict[str, nn.Module] = {}
        self.queued_tasks_by_name = collections.defaultdict(list)

        self.inference_example_count = 0
        self.inference_count = 0

        # These are the only variables accessed from multiple threads.
        self.communication_queue = collections.deque()
        self.communication_event = threading.Event()
        self.done_event = threading.Event()

        self.inference_thread_lock = threading.Lock()

    def set_nets(self, nets: Dict[str, nn.Module]):
        self.nets = nets
        for name, net in nets.items():
            net.to(self.device)

    @rpc.functions.async_execution
    def infer(self, net_name: str, args):
        future = rpc.Future()
        self.communication_queue.append((net_name, future, args))
        self.communication_event.set()
        return future

    def _unload_communication_queue(self):
        # print("unloading queue size {}".format(len(self.communication_queue)))
        while self.communication_queue:
            net_name, future, args = self.communication_queue.popleft()
            self.queued_tasks_by_name[net_name].append((future, args))
        # print("queued task size  {} {}".format(len(self.queued_tasks_by_name), sum([len(v) for k, v in self.queued_tasks_by_name.items()])))

    def _tensorize_batch(self, batch: List[Tuple[State, EncodedActionSet, Optional[List[Action]]]]) -> (StateBatch, EncodedActionSet):
        device = self.device
        player_tensor = torch.cat([b[0].player_tensor for b in batch], dim=0).detach()
        cards_tensor = torch.cat([b[0].cards_tensor for b in batch], dim=0).detach()
        valid_player_actions_tensor = torch.cat(
            [b[1].player_action_tensor for b in batch], dim=0).detach()
        valid_card_actions_tensor = torch.cat(
            [b[1].card_action_tensor for b in batch], dim=0).detach()
        rearrange_phase = torch.cat([b[1].rearrange_phase for b in batch], dim=0).detach()
        cards_to_rearrange = torch.cat(
            [b[1].cards_to_rearrange for b in batch], dim=0).detach()
        chosen_actions = None if batch[0][2] is None else [b[2] for b in batch]
        return (StateBatch(player_tensor=player_tensor.to(device),
                           cards_tensor=cards_tensor.to(device)),
                EncodedActionSet(player_action_tensor=valid_player_actions_tensor.to(device),
                                 card_action_tensor=valid_card_actions_tensor.to(device),
                                 rearrange_phase=rearrange_phase.to(device),
                                 cards_to_rearrange=cards_to_rearrange.to(device)),
                chosen_actions,
                )

    def _worker_thread(self):
        while True:
            with self.inference_thread_lock:
                self.communication_event.clear()
                self._unload_communication_queue()
                # Select the longest queue
                if self.queued_tasks_by_name:
                    net_name, _ = max(self.queued_tasks_by_name.items(),
                                 key=lambda kv: len(kv[1]))
                    tasks = self.queued_tasks_by_name.pop(net_name)
                    # Remove the first batch worth from the net specific queue
                    length = min(len(tasks), self.max_batch_size)
                    batched_tasks = [tasks.pop() for _ in range(length)]
                    self.queued_tasks_by_name[net_name] += tasks
                else:
                    length = 0
            if length:
                self.inference_count += 1
                self.inference_example_count += len(batched_tasks)
                print("Inference {} {} {}".format(self.inference_count, len(batched_tasks), float(self.inference_example_count)/ self.inference_count))

                # Run inference on batched tensor
                state_batch, valid_actions_batch, chosen_actions_batch = self._tensorize_batch([args for _, args in batched_tasks])
                net = self.nets[net_name]
                output_actions, action_log_probs, value, debug_info = net(state_batch, valid_actions_batch, chosen_actions_batch)
                cpu_device = torch.device('cpu')
                for i, (future, _) in enumerate(batched_tasks):
                    future.set_result((output_actions[i:i+1],
                                action_log_probs[i:i+1].detach().to(cpu_device),
                                value[i:i+1].detach().to(cpu_device),
                                ActorCriticGameStepDebugInfo(
                                    component_policy=debug_info.component_policy[i:i+1].detach().to(cpu_device),
                                    permutation_logits=debug_info.permutation_logits[i:i+1].detach().to(cpu_device),
                                )
                                ))
            self.communication_event.wait(1)
            if self.done_event.is_set():
                return

    def start_worker_thread(self):
        for _ in range(self.num_inference_threads):
            inference_thread = threading.Thread(target=self._worker_thread)
            inference_thread.start()

    def kill_worker_thread(self):
        self.done_event.set()
