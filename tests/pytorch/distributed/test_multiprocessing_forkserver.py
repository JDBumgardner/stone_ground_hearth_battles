import logging
import unittest

from torch import multiprocessing

from hearthstone.testing.battlegrounds_test_case import BattleGroundsTestCase


def test_fn(rank):
    print(f"hello world {rank}")


class PytorchDistributedTests(BattleGroundsTestCase):
    def test_multiprocessing_forkserver(self):
        process_context = multiprocessing.start_processes(
            test_fn,
            nprocs=4,
            start_method="forkserver",
            join=False
        )
        process_context.join()


if __name__ == '__main__':
    unittest.main()
