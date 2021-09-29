import os
import unittest

from torch import multiprocessing
from torch.distributed import rpc
from torch.distributed.rpc import RRef


def rpc_backed_options():
    device_maps = {
        'simulator': {0: 0},
        'inference': {0: 0}
    }
    return rpc.TensorPipeRpcBackendOptions(device_maps=device_maps)


def run_worker(rank, num_workers: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    rpc.init_rpc('simulator', rank=rank + 1, world_size=num_workers + 1, rpc_backend_options=rpc_backed_options())
    rpc.shutdown()


class ContainsRRef:
    def __init__(self, rref):
        self.rref = rref

    def foo(self):
        pass


class PytorchDistributedTests(unittest.TestCase):

    def test_rpc_with_rref(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

        self.process_context = multiprocessing.start_processes(
            run_worker,
            args=(1,),
            nprocs=1,
            start_method="forkserver",
            join=False
        )
        local_object = {}
        rpc.init_rpc('inference', rank=0, world_size=2, rpc_backend_options=rpc_backed_options())
        sim_info = rpc.get_worker_info('simulator')
        remote_object = rpc.remote(sim_info, ContainsRRef, args=(RRef(local_object),))
        remote_object.rpc_async().foo()


if __name__ == '__main__':
    unittest.main()
