import logging
import os
import platform
from typing import List

import torch
from torch import multiprocessing
from torch.distributed import rpc
# Note that the simulators may do the inference instead, if running in non-batched mode.
if platform.system() != 'Windows':
    from torch.distributed.rpc import RRef

from hearthstone.ladder.ladder import Contestant
from hearthstone.training.pytorch.agents.pytorch_bot import PytorchBot
from hearthstone.training.pytorch.worker.distributed.inference_worker import InferenceWorker
from hearthstone.training.pytorch.worker.distributed.remote_net import RemoteNet
from hearthstone.training.pytorch.worker.distributed.simulation_worker import SimulationWorker
from hearthstone.training.pytorch.worker.postprocessing import ReplaySink

logger = logging.getLogger(__name__)

INFERENCE_PROCESS_NAME = "inferrer"
SIMULATOR_PROCESS_NAMES = "simulator_{}"


def rpc_backed_options(num_workers):
    device_maps = {
        SIMULATOR_PROCESS_NAMES.format(i): {0: 0} for i in range(num_workers)
    }
    device_maps.update({INFERENCE_PROCESS_NAME: {0: 0}})
    return rpc.TensorPipeRpcBackendOptions(device_maps=device_maps)


def run_worker(rank, num_workers: int):
    logging.basicConfig(level=logging.WARN)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    rpc.init_rpc(SIMULATOR_PROCESS_NAMES.format(rank), rank=rank + 1, world_size=num_workers + 1,
                 rpc_backend_options=rpc_backed_options(num_workers))
    rpc.shutdown()


class DistributedWorkerPool:
    def __init__(self, num_workers: int,
                 games_per_worker: int,
                 use_batched_inference: bool,
                 max_batch_size: int,
                 replay_sink: ReplaySink,
                 device: torch.device,
                 ):
        self.num_workers = num_workers
        self.games_per_worker = games_per_worker
        self.batched = use_batched_inference
        self.replay_sink = replay_sink
        self.device = device

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

        logger.info(f"Launching {num_workers} workers.")
        self.process_context = multiprocessing.start_processes(
            run_worker,
            args=(self.num_workers,),
            nprocs=self.num_workers,
            start_method="forkserver",
            join=False
        )
        rpc.init_rpc(INFERENCE_PROCESS_NAME, rank=0, world_size=num_workers + 1,
                     rpc_backend_options=rpc_backed_options(self.num_workers))

        if self.batched:
            self.inference_worker = InferenceWorker(max_batch_size, 1, device)
            self.inference_worker.start_worker_thread()
        else:
            self.inference_worker = None

        self.simulator_rrefs: List[RRef] = []
        for sim_rank in range(self.num_workers):
            sim_info = rpc.get_worker_info(SIMULATOR_PROCESS_NAMES.format(sim_rank))
            self.simulator_rrefs.append(rpc.remote(sim_info, SimulationWorker, args=(RRef(self.inference_worker),)))

    def play_games(self, learning_bot_contestant: Contestant, other_contestants: List[Contestant], game_size: int):
        all_contestants = [learning_bot_contestant] + other_contestants
        nets = {}
        devices = {}
        for contestant in all_contestants:
            if contestant.agent_generator.function == PytorchBot:
                nets[contestant.name] = contestant.agent_generator.kwargs['net']
                devices[contestant.name] = contestant.agent_generator.kwargs['device']
                contestant.agent_generator.kwargs['net'] = RemoteNet(contestant.name, RRef(self.inference_worker))
                contestant.agent_generator.kwargs['device'] = torch.device('cpu')
        self.inference_worker.set_nets(nets)
        futures = []
        for sim_rref in self.simulator_rrefs:
            futures.append(sim_rref.rpc_async(timeout=120000).play_interleaved_games(self.games_per_worker,
                                                                                     learning_bot_contestant,
                                                                                     other_contestants, game_size))
        crash = None
        for future in futures:
            try:
                replays = future.wait()
            except Exception as e:
                crash = e
                continue
            for replay in replays:
                self.replay_sink.process(replay, learning_bot_contestant, other_contestants)
        for contestant in all_contestants:
            if contestant.agent_generator.function == PytorchBot:
                contestant.agent_generator.kwargs['net'] = nets[contestant.name]
                contestant.agent_generator.kwargs['device'] = devices[contestant.name]
        if crash:
            raise crash

    def shutdown(self):
        if self.batched:
            self.inference_worker.kill_worker_thread()
        rpc.shutdown()
        self.process_context.join()
