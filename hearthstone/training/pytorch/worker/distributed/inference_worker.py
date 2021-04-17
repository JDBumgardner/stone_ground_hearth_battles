import collections
import logging
import threading
import time
from typing import Dict

import torch
from torch import nn
from torch.distributed import rpc

from hearthstone.training.pytorch.worker.distributed.tensorize_batch import _tensorize_batch, _untensorize_batch

logger = logging.getLogger(__name__)


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
        logger.debug("unloading queue size {}".format(len(self.communication_queue)))
        while self.communication_queue:
            net_name, future, args = self.communication_queue.popleft()
            self.queued_tasks_by_name[net_name].append((future, args))
        logger.debug("queued task size  {} {}".format(len(self.queued_tasks_by_name),
                                                      sum([len(v) for k, v in self.queued_tasks_by_name.items()])))

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
                # Run inference on batched tensor
                batch_args = [args for _, args in batched_tasks]
                t = time.time()
                state_batch, valid_actions_batch, chosen_actions_batch = _tensorize_batch(batch_args
                                                                                          , self.device)
                self.inference_count += 1
                self.inference_example_count += state_batch[0].shape[0]

                logger.debug("Inference #{}: {} requests, {} total batch size, {} average batch size".format(
                    self.inference_count, len(batched_tasks),
                    state_batch[0].shape[0],
                    float(self.inference_example_count) / self.inference_count))

                net = self.nets[net_name]
                output_actions, action_log_probs, value, debug_info = net(state_batch, valid_actions_batch,
                                                                          chosen_actions_batch)
                for (future, _), unbatched in zip(
                        batched_tasks,
                        _untensorize_batch(batch_args, output_actions, action_log_probs, value, debug_info,
                                           torch.device('cpu'))):
                    future.set_result(unbatched)
                logger.debug(f"Time taken is {time.time() - t}")
            self.communication_event.wait(1)
            if self.done_event.is_set():
                return

    def start_worker_thread(self):
        for _ in range(self.num_inference_threads):
            inference_thread = threading.Thread(target=self._worker_thread)
            inference_thread.start()

    def kill_worker_thread(self):
        self.done_event.set()
