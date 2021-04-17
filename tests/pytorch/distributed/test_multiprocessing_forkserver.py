import logging
import unittest

from torch import multiprocessing

logging.basicConfig(level=logging.DEBUG)


def test_fn(rank):
    print(f"hello world {rank}")


class PytorchDistributedTests(unittest.TestCase):
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
