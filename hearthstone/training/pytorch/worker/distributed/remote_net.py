import asyncio


class RemoteNet:
    def __init__(self, net_name: str, inference_queue):
        self.net_name = net_name
        self.inference_queue = inference_queue

    async def __call__(self, *args):
        loop = asyncio.get_event_loop()
        f = loop.create_future()
        self.inference_queue.rpc_async().infer(self.net_name, args).then(lambda fut:
                                                                         loop.call_soon_threadsafe(
                                                                             f.set_result, fut.value()
                                                                         )
                                                                         )
        return await f
