import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from utils import DifferentiableBernoulli


class Dropout(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        pass


class FixedDropout(Dropout):
    def __init__(self, p=.2, tau=1, soft=False):
        # p can betensor, should be inverse sigmoid of dropout prob
        super().__init__()
        self.p = p
        self.tau = tau
        self.soft = soft

    def forward(self, x, **kwargs):
        if not self.training:
            return x
        else:
            if self.p.ndim != 0:
                assert self.p.ndim == 1 and self.p.shape[0] == x.shape[1]
            db = DifferentiableBernoulli(probs=1 - torch.sigmoid(self.p), tau=self.tau)
            mask = db.sample(shape=x.shape, soft=self.soft)
            return mask * x / (1 - torch.sigmoid(self.p))


class FixedDropoutPerLayer(FixedDropout):
    def __init__(self, layer, p=None, tau=1, soft=False):
        super().__init__(p, tau, soft)
        p = p if p is not None else [.2]
        self.p = p[layer]


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
