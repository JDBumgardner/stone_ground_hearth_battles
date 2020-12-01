import time

import torch
from torch import nn
import torch.nn.functional as F

from hearthstone.battlebots.no_action_bot import NoActionBot
from hearthstone.simulator.host.round_robin_host import RoundRobinHost
from hearthstone.training.pytorch.encoding.default_encoder import DEFAULT_PLAYER_ENCODING, DEFAULT_CARDS_ENCODING, \
    DefaultEncoder
from hearthstone.training.pytorch.networks.transformer_net import HearthstoneTransformerNet
from hearthstone.training.pytorch.pytorch_bot import PytorchBot


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(50, 50)
        self.layer2 = nn.Linear(50, 50)

    def forward(self, input):
        return self.layer2(F.relu(self.layer1(input)))


def process(net: MyNet, tensor: torch.Tensor):
    with torch.no_grad():
        begin_time = time.time()
        for _ in range(80000):
            tensor = net(tensor)
        end_time = time.time()
        print("process time ", end_time - begin_time)
        result = float(tensor[0])
        return result


def process_hearthstone():
    # set_num_threads is important here because openMP messes things up.
    torch.set_num_threads(1)
    with torch.no_grad():
        host = RoundRobinHost(
            {"Bot1": PytorchBot(HearthstoneTransformerNet(DefaultEncoder(),
                                                          hidden_layers=1, hidden_size=32), DefaultEncoder(), True),
             "Bot2": NoActionBot()},
            []
        )
        print("Beginning!")
        start = time.time()
        host.start_game()
        for i in range(20):
            host.play_round()
        print("Done!", time.time() - start)


def run_single_process_benchmark(num_workers: int):
    with torch.no_grad():
        net = MyNet()
        tensor = torch.rand((50,))
        begin_time = time.time()
        results = [
            process_hearthstone()
            for _ in
            range(num_workers)]
        end_time = time.time()
        print("total time ", end_time - begin_time)


def run_multiprocess_benchmark(num_workers: int):
    with torch.no_grad():
        net = MyNet()
        net.share_memory()
        tensor = torch.rand((50,))
        tensor.share_memory_()
        pool = torch.multiprocessing.Pool(processes=num_workers)

        begin_time = time.time()
        awaitables = [
            pool.apply_async(process_hearthstone, ())
            for _ in
            range(num_workers)]
        for promise in awaitables:
            promise.get()
        end_time = time.time()
        print("total time ", end_time - begin_time)


def main():
    num_workers = 2
    run_single_process_benchmark(num_workers)
    run_multiprocess_benchmark(num_workers)


if __name__ == '__main__':
    main()
