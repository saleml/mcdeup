import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from old.utils import DifferentiableBernoulli


class Dropout(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        pass


class RegularDropout(Dropout):
    def __init__(self, logit=None, tau=1, soft=False, noise_generator=None):
        # logit should be a Parameter tensor of no dimension or of dimension 1
        # noise_generator should be of instance nn.Module, its output dimension should be equal to the hidden size
        # or to 1
        super().__init__()
        self.noise_generator = noise_generator
        if logit is None:
            logit = nn.Parameter(torch.tensor(-2.))
        self.tau = tau
        self.soft = soft
        if isinstance(logit, nn.Parameter):
            self.register_parameter('logit', logit)

    def forward(self, x, parametric_noise=False, noise_input=None):
        if not self.training:
            return x
        else:
            if parametric_noise:
                assert self.noise_generator is not None and isinstance(self.noise_generator, nn.Module)
                noise_input = x if noise_input is None else noise_input
                logit = self.noise_generator(noise_input)
            else:
                logit = self.logit
            # if logit.ndim != 0:
            #    assert logit.ndim == 1 and logit.shape[0] == x.shape[1]
            db = DifferentiableBernoulli(probs=1-torch.sigmoid(logit), tau=self.tau)
            mask = db.sample(shape=x.shape, soft=self.soft)
            return mask * x / (1 - torch.sigmoid(logit))


class RegularDropoutPerLayer(RegularDropout):
    def __init__(self, layer, logit=None, tau=1, soft=False, noise_generator=None):
        # logit should be a list of Parameters of dimension 0 or dimension 1 corresponding to the hidden size,
        # and noise_generator a list of nn.Modules
        super().__init__(logit, tau, soft)
        self.logit = logit[layer] if logit is not None else nn.Parameter(torch.tensor(-2.))
        if noise_generator is not None:
            assert isinstance(noise_generator, list)
            self.noise_generator = noise_generator[layer]


class MultiplicativeGaussian(Dropout):
    def __init__(self, mu=None, logsigma=None, noise_generator=None):
        # mu and sigma should be Parameter tensors of no dimension or of dimension 1 corresponding to the hidden size
        # noise_generator should be of instance nn.Module, its output dimension should be equal to the hidden size
        # or to 1
        super().__init__()
        self.noise_generator = noise_generator
        if mu is None:
            mu = nn.Parameter(torch.tensor(1.))
        if logsigma is None:
            logsigma = nn.Parameter(torch.tensor(-2.))
        if isinstance(mu, nn.Parameter) and isinstance(logsigma, nn.Parameter):
            self.register_parameter('mu', mu)
            self.register_parameter('logsigma', logsigma)

    def forward(self, x, parametric_noise=False, noise_input=None):
        if not self.training:
            return x
        else:
            if parametric_noise:
                assert self.noise_generator is not None and isinstance(self.noise_generator, nn.Module)
                noise_input = x if noise_input is None else noise_input
                logsigma = self.noise_generator(noise_input)
            else:
                logsigma = self.logsigma
            noise = torch.exp(logsigma) * torch.randn_like(x) + self.mu
            return noise * x / self.mu


class MultiplicativeGaussianPerLayer(MultiplicativeGaussian):
    def __init__(self, layer, mu=None, logsigma=None, noise_generator=None):
        # mu and sigma should be lists of Parameters, and noise_generator a list of nn.Modules
        super().__init__(mu, logsigma)
        self.mu = mu[layer] if mu is not None else nn.Parameter(torch.tensor(1.))
        self.logsigma = logsigma[layer] if logsigma is not None else nn.Parameter(torch.tensor(-2.))
        if noise_generator is not None:
            assert isinstance(noise_generator, list)
            self.noise_generator = noise_generator[layer]
