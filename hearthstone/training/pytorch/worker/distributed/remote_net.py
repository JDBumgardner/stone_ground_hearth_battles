import asyncio
import logging
import os

import torch

from hearthstone.training.pytorch.worker.distributed.tensorize_batch import _tensorize_batch, _untensorize_batch

logger = logging.getLogger(__name__)


class RemoteNet:
    def __init__(self, net_name: str, inference_queue):
        self.net_name = net_name
        self.inference_queue = inference_queue

    async def __call__(self, *args):
        loop = asyncio.get_event_loop()
        f = loop.create_future()
        self.inference_queue.rpc_async().infer(self.net_name, args).add_done_callback(lambda fut:
                                                                                      loop.call_soon_threadsafe(
                                                                                          f.set_result, fut.value()
                                                                                      )
                                                                                      )
        return await f


class BatchedRemoteNet:
    def __init__(self, net_name: str, inference_queue):
        self.net_name = net_name
        self.inference_queue = inference_queue

        self.unbatched_requests_event = asyncio.Event()
        self.unbatched_requests = []
        self.active_rpc = None
        self._stop_worker = False
        self._worker_task = None

    async def __call__(self, *args):
        loop = asyncio.get_event_loop()
        f = loop.create_future()
        self.unbatched_requests.append((args, f))
        self.unbatched_requests_event.set()
        return await f

    async def start_worker(self):
        assert self._worker_task is None
        self._stop_worker = False
        self._worker_task = asyncio.create_task(self.worker_task())

    async def stop_worker(self):
        self._stop_worker = True
        self.unbatched_requests_event.set()
        await self._worker_task

    async def worker_task(self):
        while True:
            await self.unbatched_requests_event.wait()
            if self._stop_worker:
                return
            self.unbatched_requests_event.clear()
            unbatched_futures = [b[1] for b in self.unbatched_requests]
            unbatched_args = [b[0] for b in self.unbatched_requests]
            self.unbatched_requests.clear()
            args = _tensorize_batch(unbatched_args, torch.device('cpu'))
            loop = asyncio.get_event_loop()
            f = loop.create_future()

            logger.debug(f"Calling RPC {os.getpid()}")
            self.inference_queue.rpc_async().infer(self.net_name, args).add_done_callback(lambda fut:
                                                                                          loop.call_soon_threadsafe(
                                                                                              f.set_result, fut.value()
                                                                                          )
                                                                                          )
            response = await f
            logger.debug(f"RPC complete {os.getpid()}")
            for future, result in zip(
                    unbatched_futures,
                    _untensorize_batch(unbatched_args,
                                       *response,
                                       torch.device('cpu'))):
                future.set_result(result)
