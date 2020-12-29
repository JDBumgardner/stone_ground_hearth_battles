import unittest
from typing import Optional, Tuple

import torch

from hearthstone.training.pytorch.networks.plackett_luce import PlackettLuce


class PlackettLuceTest(unittest.TestCase):
    def assertIsPermutation(self, sample: torch.Tensor, shape: Tuple, permutation_sizes: Optional[torch.Tensor] = None):
        self.assertEqual(sample.shape, shape)
        self.assertTrue((sample.sort(-1).values == torch.arange(0, sample.shape[-1])).all())
        if permutation_sizes is not None:
            indices = (permutation_sizes.unsqueeze(-1).expand((*sample.shape[:-1], 1)) - 1)
            self.assertTrue(torch.eq(sample.cumsum(-1).gather(-1, indices).squeeze(-1),
                                     permutation_sizes * (permutation_sizes - 1) // 2).all())

    def test_single(self):
        logits = torch.Tensor([[10.0, 20.0, 30.0, 40.0]])
        permutation_sizes = torch.LongTensor([3])
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = distribution.sample()
        self.assertIsPermutation(sample, (1, 4), permutation_sizes)
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, logits.shape[:-1])

    def test_max_size(self):
        logits = torch.Tensor([[10.0, 20.0, 30.0, 40.0]])
        permutation_sizes = torch.LongTensor([4])
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = distribution.sample()
        self.assertIsPermutation(sample, (1, 4), permutation_sizes)
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, logits.shape[:-1])

    def test_default_sizes(self):
        logits = torch.Tensor([[10.0, 20.0, 30.0, 40.0]])
        distribution = PlackettLuce(logits)
        sample = distribution.sample()
        self.assertIsPermutation(sample, (1, 4))
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, logits.shape[:-1])

    def test_batched(self):
        logits = torch.Tensor([[10.0, 20.0, 30.0, 40.0],
                               [1, 1, 5, 4],
                               [5, 6, 6, 6]])
        permutation_sizes = torch.LongTensor([3, 4, 2])
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = distribution.sample()
        self.assertIsPermutation(sample, (3, 4), permutation_sizes)
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, logits.shape[:-1])

    def test_shaped_sample(self):
        logits = torch.Tensor([[10.0, 20.0, 30.0, 40.0]])
        permutation_sizes = torch.LongTensor([3])
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = distribution.sample((10,))
        self.assertIsPermutation(sample, (10, 1, 4), permutation_sizes)
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, (10, *logits.shape[:-1]))

    def test_batched_shaped_sample(self):
        logits = torch.Tensor([[10.0, 20.0, 30.0, 40.0],
                               [1.0, 2.0, 3.0, 4.0],
                               [0.0, 0.1, 0.6, -0.5],
                               [0.0, 2.0, 0.0, 1.5]])
        permutation_sizes = torch.LongTensor([3, 4, 4, 2])
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = distribution.sample((10,))
        self.assertIsPermutation(sample, (10, 4, 4), permutation_sizes)
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, (10, *logits.shape[:-1]))


if __name__ == '__main__':
    unittest.main()
