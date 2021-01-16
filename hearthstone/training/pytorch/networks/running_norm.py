import torch
from torch import nn

from hearthstone.training.pytorch.encoding.state_encoding import Encoder, State


class WelfordAggregator:
    def __init__(self, shape:torch.Size):
        self.shape = shape
        self.count = 0
        self.mean = torch.zeros(shape)
        self.m2 = torch.zeros(shape)

    def update(self, value: torch.Tensor):
        with torch.no_grad():
            value = value.reshape((-1,) + self.shape)
            b_count = value.shape[0]
            b_mean = value.mean(dim=0)
            b_m2 = (value - b_mean).pow(2).sum(dim=0)
            n = self.count + b_count
            delta = b_mean - self.mean

            self.m2 = self.m2 + b_m2 + delta.pow(2) * self.count * b_count / n
            self.mean += delta * b_count / n
            self.count = n

    def mean(self):
        return self.mean

    def variance(self):
        with torch.no_grad():
            return self.m2 / self.count

    def stdev(self):
        with torch.no_grad():
            return torch.sqrt(self.variance())


class PPONormalizer(nn.Module):
    def __init__(self, shape: torch.Size, gamma: float, epsilon: float = 1e-5):
        """
        This is the reward normalization scheme defined in https://openreview.net/pdf?id=r1etN1rtPB, Appendix A2.

        Note that it updates in eval mode but does not update in train mode, which is the opposite of a batch-norm layer.
        Args:
            shape (tuple): Shape of the observation tensor that we're normalizing.
            gamma (float): The running mean
        """
        super().__init__()
        self.shape = shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.exponential_mean = torch.zeros(shape)
        self.welford_aggregator = WelfordAggregator(shape)

    def forward(self, value: torch.Tensor):
        if not self.training:
            with torch.no_grad():
                flattened = value.reshape((-1,) + self.shape)
                num_updates = flattened.shape[0]
                coefficients = torch.pow(self.gamma, torch.arange(num_updates)).view((num_updates,) + (1,) * (len(flattened.shape) -1) )
                self.exponential_mean = self.gamma ** num_updates * self.exponential_mean + flattened * coefficients
                self.welford_aggregator.update(self.exponential_mean)
        if self.welford_aggregator.count > 2:
            return value / (self.welford_aggregator.stdev() + self.epsilon)
        else:
            return value

class EMANormalizer(nn.Module):
    def __init__(self, shape: torch.Size, gamma: float, epsilon: float = 1e-5):
        """
        This is a raw normalizer which normalizes by a running mean/stddev.

        Note that it updates in eval mode but does not update in train mode, which is the opposite of a batch-norm layer.
        Args:
            shape (tuple): Shape of the observation tensor that we're normalizing.
            gamma (float): The running mean
        """
        super().__init__()
        self.shape = shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.exponential_mean = torch.zeros(shape)
        self.welford_aggregator = WelfordAggregator(shape)

    def forward(self, value: torch.Tensor):
        if not self.training:
            self.welford_aggregator.count *= self.gamma
            self.welford_aggregator.update(self.value)
        if self.welford_aggregator.count > 2:
            return value - self.welford_aggregator.mean() / (self.welford_aggregator.stdev() + self.epsilon)
        else:
            return value


class ObservationNormalizer(nn.Module):
    def __init__(self, encoding: Encoder, gamma: float):
        super().__init__()
        self.player_normalizer = EMANormalizer(torch.Shape(encoding.player_encoding().size()), gamma)
        self.cards_normalizer = EMANormalizer(torch.Shape(encoding.cards_encoding().size()[1:]), gamma)

    def forward(self, state: State):
        return State(player_tensor=self.player_normalizer(state.player_tensor),
                     cards_tensor=self.cards_normalizer(state.cards_tensor)
                     )
