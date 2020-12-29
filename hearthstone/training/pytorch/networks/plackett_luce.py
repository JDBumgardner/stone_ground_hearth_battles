from typing import Optional

import torch
from torch.distributions import Distribution, constraints


class PlackettLuce(Distribution):
    arg_constraints = {'probs': constraints.real}
    has_enumerate_support = True

    def __init__(self, logits: torch.Tensor, permutation_sizes: Optional[torch.Tensor] = None, validate_args=None):
        batch_shape = logits.size()[:-1]
        super(PlackettLuce, self).__init__(batch_shape, validate_args=validate_args)
        self.logits: torch.Tensor = logits

        if permutation_sizes is None:
            permutation_sizes = torch.full(self.logits.shape[:-1], self.logits.shape[-1], dtype=torch.int64, device=logits.device)
        self.permutation_sizes: torch.Tensor = permutation_sizes
        # Mask is true for invalid indices
        with torch.no_grad():
            self.mask: torch.Tensor = torch.zeros(
                *logits.shape[:-1], logits.shape[-1] + 1, device=logits.device).scatter(-1,
                                                                  permutation_sizes.unsqueeze(-1),
                                                                  1)[...,
                                      :-1].cumsum(
                dim=-1).bool()

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        with torch.no_grad():
            expanded = self.logits.expand(*sample_shape, * [-1] * len(self.logits.shape))
            gumbel_noise = - torch.log(-torch.log(torch.rand_like(expanded)))
            scores = (expanded + gumbel_noise) + -1e30 * self.mask
            sorted_scores, indices = torch.sort(scores, dim=-1, descending=True)
            return indices

    def log_prob(self, value: torch.Tensor):
        logits = self.logits.masked_fill(self.mask, -1e30).expand(value.shape)
        log_probs = torch.zeros(value.shape[:-1], device=value.device)
        for i in range(int(self.permutation_sizes.max())):
            log_probs += logits.log_softmax(dim=-1).gather(-1, value[..., i:i+1]).squeeze(-1) * self.mask.logical_not()[..., i]
            logits = logits.scatter(-1, value[..., i:i + 1], -1e30)
        return log_probs

