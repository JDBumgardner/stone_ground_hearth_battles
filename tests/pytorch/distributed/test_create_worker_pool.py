import logging
import unittest

import torch

from hearthstone.training.pytorch.worker.distributed.worker_pool import DistributedWorkerPool
from hearthstone.training.pytorch.worker.postprocessing import ReplaySink

logging.basicConfig(level=logging.DEBUG)


class PytorchDistributedTests(unittest.TestCase):

    def test_create_worker_pool(self):
        p = DistributedWorkerPool(5,
                                  1,
                                  True,
                                  1024,
                                  ReplaySink(),
                                  torch.device('cpu'),
                                  )
        p.shutdown()


if __name__ == '__main__':
    unittest.main()
