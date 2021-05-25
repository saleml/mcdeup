import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from torch.distributions import Normal


class NoiseGenerator(nn.Module, ABC):
    """
    Base class for the different noise generator types
    """
    def __init__(self, size):
        """
        :param size: dimension of the noise space (can be as large as the number of neurons)
        """
        super().__init__()
        self.size = size

    @abstractmethod
    def forward(self, x):
        """
        :param x: input of the noise generator, tensor of shape B x ...
        :return: parameters of the distribution, tensor(s) of shape B x ...
        """
        pass

    @abstractmethod
    def sample(self, x, fixed_noise=False):
        """
        :param x: input of the noise generator, tensor of shape B x ...
        :param fixed_noise: if True, then the same sample is used for the whole batch
        :return: samples from the distribution parametrized by self(x), of shape n_samples x B x size
        """
        pass


class GaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, size, mu=None, logsigma=None, model=None, model_out='both'):
        # For mu and sigma, if they are None, then they are both learned, else they are the constant tensors passed as arguments
        # model_out is a string: 'both', 'mu', 'logsigma'
        super().__init__(size)
        self.model = model
        self.model_out = model_out
        if mu is None and (model is None or model_out == 'logsigma'):
            self.mu = nn.Parameter(torch.zeros(size))
        else:
            self.mu = mu
        if logsigma is None and (model is None or model_out == 'mu'):
            self.logsigma = nn.Parameter(-2. * torch.ones(size))
        else:
            self.logsigma = logsigma

    def forward(self, x):
        if self.model is None:
            return self.mu.repeat(x.shape[0], 1), self.logsigma.repeat(x.shape[0], 1)

        out = self.model(x)
        assert out.shape[0] == x.shape[0], out.shape[1] == (2 if self.model_out == 'both' else 1) * self.size
        if self.model_out == 'both':
            mu, logsigma = out[:, :self.size], out[:, self.size:]
        elif self.model_out == 'mu':
            mu = out
            logsigma = self.logsigma.repeat(x.shape[0], 1)
        elif self.model_out == 'logsigma':
            mu = self.mu.repeat(x.shape[0], 1)
            logsigma = out
        return mu, logsigma

    def sample(self, x, fixed_noise=False):
        mu, logsigma = self(x)  # of shape B x size each
        samples = mu + torch.exp(logsigma) * torch.randn(*mu.shape)
        if fixed_noise:
            return samples[0, :].repeat(mu.shape[0], 1)
        return samples


class CauchyNoiseGenerator(NoiseGenerator):
    pass

