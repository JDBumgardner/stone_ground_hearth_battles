import torch


class WelfordAggregator:
    def __init__(self, shape):
        self.count = 0
        self.mean = torch.zeros(shape)
        self.m2 = torch.zeros(shape)

    def update(self, value):
        value = value.detach()
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def mean(self):
        return self.mean

    def variance(self):
        return self.m2 / self.count

    def stdev(self):
        return torch.sqrt(self.variance())


class PPONormalizer:
    def __init__(self, gamma: float, shape: tuple):
        """
        This is the reward normalization scheme defined in https://openreview.net/pdf?id=r1etN1rtPB, Appendix A2.
        Args:
            gamma (float): The reward discount
            shape (tuple): Shape of the observation tensor that we're normalizing.
        """
        self.gamma = gamma
        self.exponential_mean = torch.zeros(shape)
        self.welford_aggregator = WelfordAggregator(shape)

    def normalize(self, value):
        self.exponential_mean = self.gamma*self.exponential_mean + value
        self.welford_aggregator.update(self.exponential_mean)
        if self.welford_aggregator.count > 2:
            return value / (self.welford_aggregator.stdev() + 1e-5)
        else:
            return value

