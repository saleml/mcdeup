import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from utils import DifferentiableBernoulli

import math

class Dropout(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        pass


class FixedDropout(Dropout):
    def __init__(self, p=None, tau=1, soft=False):
        # p should be a Parameter tensor of no dimension or of dimension 1, should be inverse sigmoid of dropout prob
        if p is None:
            p = nn.Parameter(torch.tensor(-2))
        super().__init__()
        self.tau = tau
        self.soft = soft
        if isinstance(p, nn.Parameter):
            self.register_parameter('p', p)

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
        # p should be a list of Parameters of dimension 0 or dimension 1 corresponding to the hidden size
        super().__init__(p, tau, soft)
        self.register_parameter('p', p[layer])


class FixedMultiplicativeGaussian(Dropout):
    def __init__(self, mu=None, logsigma=None):
        # mu and sigma should be Parameter tensors of no dimension or of dimension 1 corresponding to the hidden size
        super().__init__()
        if mu is None:
            mu = nn.Parameter(torch.tensor(1))
        if logsigma is None:
            logsigma = nn.Parameter(torch.tensor(-2))
        if isinstance(mu, nn.Parameter) and isinstance(logsigma, nn.Parameter):
            self.register_parameter('mu', mu)
            self.register_parameter('logsigma', logsigma)

    def forward(self, x, **kwargs):
        if not self.training:
            return x
        else:
            noise = torch.exp(self.logsigma) * torch.randn_like(x) + self.mu
            return noise * x / self.mu


class FixedMultiplicativeGaussianPerLayer(FixedMultiplicativeGaussian):
    def __init__(self, layer, mu=None, logsigma=None):
        # mu and sigma should be lists of Parameters
        super().__init__(mu, logsigma)
        self.mu = mu[layer]
        self.logsigma = logsigma[layer]


class LearnedDropout(Dropout):
    def __init__(self, p=.2, tau=1, soft=False, **kwargs):
        # p can betensor, should be inverse sigmoid of dropout prob
        super().__init__()
        self.p = p
        self.tau = tau
        self.soft = soft

    def forward(self, x, parametric_noise=False, noise_input=None, **kwargs):
        if not self.training:
            return x
        else:
            if parametric_noise:
                assert isinstance(self.p, nn.Module)
                print(noise_input)
                p = self.p(noise_input)
                assert p.shape == x.shape
                db = DifferentiableBernoulli(probs=1 - torch.sigmoid(p), tau=self.tau)
                mask = db.sample(x.shape, soft=self.soft) / (1 - p)
            else:
                db = DifferentiableBernoulli(probs=torch.tensor(.9), tau=self.tau)  # dropout = .2
                mask = db.sample(shape=x.shape, soft=self.soft) / 0.9
            return mask * x

class LearnedMultiplicativeGaussian2(Dropout):
    # dropout noise as a function of activation only
    def __init__(self, n_hidden, mu=1., logsigma=-3., **kwargs):
        # mu and sigma can be floats or tensors
        super().__init__()
        self.mu = mu
        self.logsigma = logsigma
        self.n_hidden = n_hidden

        self.noise_generator = nn.Sequential(nn.Linear(n_hidden, n_hidden))

    def forward(self, x, parametric_noise=False, noise_input=None, **kwargs):
        if not self.training:
            return x
        else:
            assert x.shape[1] == self.n_hidden
            if parametric_noise:
                logsigma = self.noise_generator(x)
            else:
                logsigma = self.logsigma
            noise = torch.exp(logsigma) * torch.randn_like(x) + self.mu
            return noise * x / self.mu





class LearnedMultiplicativeGaussian(Dropout):
    def __init__(self, n_hidden, mu=1., logsigma=-3.):
        # mu and sigma can be floats or tensors
        super().__init__()
        self.mu = mu
        self.logsigma = logsigma
        self.n_hidden = n_hidden

        self.first = nn.Linear(n_hidden, n_hidden, bias=False)
        self.second = nn.Linear(n_hidden, n_hidden, bias=False)
        self.out = nn.Linear(n_hidden, n_hidden, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.first.weight.data.normal_(0, 1/(self.n_hidden)**0.5)
        self.second.weight.data.normal_(0, 1/(2*self.n_hidden)**0.5)
        self.out.weight.data.normal_(0, math.exp(self.logsigma)/(2*self.n_hidden)**0.5)

    def forward(self, x, shared_mat=None, parametric_noise=False, noise_input=None, **kwargs):
        if not self.training:
            return x
        else:
            noise = F.relu(self.first(x))
            if shared_mat is None:
                noise = F.relu(self.second(noise))
            else:
                noise = F.relu(shared_mat.weight @ noise)

            noise = self.out(noise)
            noise = noise * torch.randn_like(x) + self.mu
            return noise * x / self.mu


class LearnedMultiplicativeGaussianPerLayer(LearnedMultiplicativeGaussian):
    def __init__(self, n_hidden=None, layer=None, mu=None, logsigma=None):
        # mu and sigma should be lists of tensors, layer an int
        super().__init__(n_hidden)
        
        self.mu = mu[layer] if mu is not None else 1.
        self.logsigma = logsigma[layer] if logsigma is not None else 1.
        self.reset_parameters()