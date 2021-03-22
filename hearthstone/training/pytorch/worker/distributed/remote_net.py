class RemoteNet:
    def __init__(self, net_name: str, inference_queue):
        self.net_name = net_name
        self.inference_queue = inference_queue

    def __call__(self, *args):
        return self.inference_queue.rpc_sync().infer(self.net_name, args)
