import unittest

import torch
from torch.distributions import Categorical

from hearthstone.simulator.core.tavern import Tavern
from hearthstone.training.pytorch.encoding.default_encoder import DefaultEncoder
from hearthstone.training.pytorch.networks.running_norm import WelfordAggregator


class PytorchTests(unittest.TestCase):
    def test_encoding(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("brian")
        tavern.buying_step()
        player_1_encoding = DefaultEncoder().encode_state(player_1)
        print(player_1_encoding)

    def test_valid_actions(self):
        tavern = Tavern()
        player_1 = tavern.add_player_with_hero("Dante_Kong")
        player_2 = tavern.add_player_with_hero("brian")
        tavern.buying_step()
        player_1_valid_actions = DefaultEncoder().encode_valid_actions(player_1, False)
        print(player_1_valid_actions)

    def test_get_stacked(self):
        tensor1 = torch.tensor([1, 2, 5, 6])
        tensor2 = torch.tensor([5, 6, 83, 7])
        print(tensor1.size())
        print(tensor1.size() + tensor2.size())
        torch.Size()

    # def test_gpu(self):
    #     tensor1 = torch.tensor([1,2,5,6])
    #     if torch.cuda.is_available():
    #         for i in range(1000):
    #             i_am_on_the_gpu = tensor1.cuda()
    #             print("put some stuff on the GPU")

    def test_sample_distribution(self):
        tensor1 = torch.tensor([[1.5, 2.3, 3.8, 4.1],
                                [0.1, 0.2, 0.3, 0.4]])
        m = Categorical(tensor1)
        samp = m.sample()
        print(samp)
        prob = tensor1.gather(1, torch.tensor([[1, 3], [2, 3]]))
        print(prob)

        other = tensor1.gather(0, torch.tensor([[0, 1, 0, 0], [1, 0, 1, 1]]))
        print(other)

    def test_welford_aggregator(self):
        agg = WelfordAggregator(torch.Size())
        data1 = torch.tensor([1., 2, 3, 4])
        data2 = torch.tensor([5., 6, 7, 8])
        data3 = torch.tensor([5., -1, -5, 8])
        agg.update(data1)
        agg.update(data2)
        agg.update(data3)
        combined = torch.cat([data1, data2, data3])
        self.assertAlmostEqual(agg.mean().item(), combined.mean().item())
        self.assertAlmostEqual(agg.stdev().item(), torch.std(combined, unbiased=False).item(), 6)


if __name__ == '__main__':
    unittest.main()
