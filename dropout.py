import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class Dropout(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        pass


class FixedDropout(Dropout):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, **kwargs):
        if not self.training:
            return x
        mask = (1 - self.p) * torch.ones_like(x)
        mask = torch.bernoulli(mask)
        return mask * x / (1 - self.p)


class FixedMultiplicativeGaussian(Dropout):
    def __init__(self, mu=1., logsigma=0.):
        # mu and sigma can be floats or tensors
        super().__init__()
        self.mu = mu
        self.logsigma = logsigma

    def forward(self, x, **kwargs):
        if not self.training:
            return x
        else:
            noise = torch.exp(self.logsigma) * torch.randn_like(x) + self.mu
            return noise * x / self.mu


class FixedMultiplicativeGaussianPerLayer(FixedMultiplicativeGaussian):
    def __init__(self, layer, mu=None, logsigma=None):
        # mu and sigma should be lists of tensors, layer an int
        super().__init__()
        mu = mu if mu is not None else [1.]
        logsigma = logsigma if logsigma is not None else [1.]
        self.mu = mu[layer]
        self.logsigma = logsigma[layer]
