import unittest
from typing import Optional, Tuple

import torch

from hearthstone.testing.battlegrounds_test_case import BattleGroundsTestCase
from plackett_luce.plackett_luce import PlackettLuce


class PlackettLuceTest(BattleGroundsTestCase):
    def assertIsPermutation(self, sample: torch.Tensor, shape: Tuple, permutation_sizes: Optional[torch.Tensor] = None):
        self.assertEqual(sample.shape, shape)

        self.assertTrue(
            (sample.masked_fill(sample == -1, sample.shape[-1]).sort(-1).values == torch.arange(0, sample.shape[-1]))
                .masked_fill(sample == -1, True).all())
        if permutation_sizes is not None and (permutation_sizes > 0).any():
            indices = (permutation_sizes.unsqueeze(-1).expand((*sample.shape[:-1], 1)) - 1)
            self.assertTrue(torch.eq(sample.cumsum(-1).gather(-1, indices).squeeze(-1),
                                     permutation_sizes * (permutation_sizes - 1) // 2).all())

    def test_single(self):
        logits = torch.Tensor([10.0, 20.0, 30.0, 40.0])
        permutation_sizes = torch.tensor(3, dtype=torch.int64)
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = distribution.sample()
        self.assertIsPermutation(sample, (4,), permutation_sizes)
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, logits.shape[:-1])

    def test_max_size(self):
        logits = torch.Tensor([10.0, 20.0, 30.0, 40.0])
        permutation_sizes = torch.tensor(3, dtype=torch.int64)
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = distribution.sample()
        self.assertIsPermutation(sample, (4,), permutation_sizes)
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, logits.shape[:-1])

    def test_default_sizes(self):
        logits = torch.Tensor([10.0, 20.0, 30.0, 40.0])
        distribution = PlackettLuce(logits)
        sample = distribution.sample()
        self.assertIsPermutation(sample, (4,))
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

    def test_size_zero(self):
        logits = torch.Tensor([])
        distribution = PlackettLuce(logits)
        sample = distribution.sample()
        self.assertEqual(sample.shape, (0,))
        log_prob = distribution.log_prob(sample)
        self.assertTrue(torch.equal(log_prob, torch.tensor(0.)))

    def test_size_double_zero(self):
        logits = torch.Tensor([[]])
        distribution = PlackettLuce(logits)
        sample = distribution.sample()
        self.assertEqual(sample.shape, (1, 0))
        log_prob = distribution.log_prob(sample)
        self.assertTrue(torch.equal(log_prob, torch.Tensor([0])))

    def test_masked_to_size_zero(self):
        logits = torch.Tensor([10.0, 20.0, 30.0, 40.0])
        permutation_sizes = torch.tensor(0, dtype=torch.int64)
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = distribution.sample()
        self.assertIsPermutation(sample, (4,), permutation_sizes)
        log_prob = distribution.log_prob(sample)
        self.assertEqual(log_prob.shape, logits.shape[:-1])

    def test_size_zero_shaped_sample(self):
        logits = torch.Tensor([])
        distribution = PlackettLuce(logits)
        sample = distribution.sample((10,))
        self.assertEqual(sample.shape, (10, 0))
        log_prob = distribution.log_prob(sample)
        self.assertTrue(torch.equal(log_prob, torch.zeros(10)))

    def test_size_double_zero_shaped_sample(self):
        logits = torch.Tensor([[]])
        distribution = PlackettLuce(logits)
        sample = distribution.sample((10,))
        self.assertEqual(sample.shape, (10, 1, 0))
        log_prob = distribution.log_prob(sample)
        self.assertTrue(torch.eq(log_prob, torch.Tensor([0] * 10)).all())

    def test_size_two_prob(self):
        logits = torch.Tensor([0, 0])
        distribution = PlackettLuce(logits)
        sample = torch.LongTensor([0, 1])
        self.assertAlmostEqual(float(distribution.log_prob(sample)), -0.693147181)

    def test_equal_scores_prob(self):
        logits = torch.Tensor([0, 0, 0, 0, 0])
        distribution = PlackettLuce(logits)
        sample = torch.LongTensor([0, 1, 2, 3, 4])
        self.assertAlmostEqual(float(distribution.log_prob(sample)), -4.787491743, 6)

    def test_equal_scores_prob_masked(self):
        logits = torch.Tensor([0, 0, 0, 0, 0, 0, 0])
        permutation_sizes = torch.tensor(5., dtype=torch.int64)
        distribution = PlackettLuce(logits, permutation_sizes)
        sample = torch.LongTensor([0, 1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(float(distribution.log_prob(sample)), -4.787491743, 6)


if __name__ == '__main__':
    unittest.main()
