import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from noise_transformer import ConstantNoiseTransformer, BlockNoiseTransformer
from noise_generator import GaussianNoiseGenerator


class NoisyNet(ABC, nn.Module):
    """
    Base class for Noisy Networks. All networks have a knob, sigma, that controls for the amount of noise
    """
    def __init__(self, input_dim, n_hidden, hidden_layers, output_dim,
                 activation_fn, noise_generator, noise_transformer):
        super().__init__()

        model = nn.Sequential()
        self.noise_generator = noise_generator
        self.noise_transformer = noise_transformer

        self.n_hidden = n_hidden

        model.add_module('input_layer', nn.Linear(input_dim, n_hidden))
        model.add_module('activation_0', activation_fn())
        for i in range(hidden_layers):
            model.add_module('hidden_layer_{}'.format(i + 1), nn.Linear(n_hidden, n_hidden))
            model.add_module('activation_{}'.format(i + 1), activation_fn())
        model.add_module('output_layer', nn.Linear(n_hidden, output_dim))
        self.model = model

    @abstractmethod
    def forward(self, x, sigma, fixed_noise=False):
        # x should be a B x d tensor
        pass

    def sample(self, X_test, fixed_noise=False):
        # X_test should be a B x d tensor
        return self(X_test, sigma=1, fixed_noise=fixed_noise)

    def get_uncertainties(self, X_test, K, fixed_noise=False):
        corrupted_outputs = torch.stack([self(X_test, sigma=1., fixed_noise=fixed_noise)
                                         for _ in range(K)])  # K x B x out tensor
        uncorrupted_output = self(X_test, sigma=0.)
        variances = (corrupted_outputs - uncorrupted_output).pow(2).mean(0)
        return variances.sqrt()


class AdditiveNoisyNet(NoisyNet):
    def __init__(self, input_dim=1, n_hidden=32, hidden_layers=1,
                 output_dim=1, activation_fn=nn.ReLU, noise_generator=None,
                 noise_transformer=None):
        if noise_generator is None:
            noise_generator = GaussianNoiseGenerator(1)
        if noise_transformer is None:
            noise_transformer = ConstantNoiseTransformer(output_dim + n_hidden * hidden_layers)
        super().__init__(input_dim, n_hidden, hidden_layers, output_dim, activation_fn, noise_generator,
                         noise_transformer)

    def forward(self, x, sigma=1., fixed_noise=False):
        z = self.noise_generator.sample(x, fixed_noise)
        n = self.noise_transformer(z)  # should be of shape B x n_units
        out = x
        i = 0
        for (name, module) in self.model.named_children():
            if 'hidden_layer' in name:
                out = module(out)
                out += sigma * n[:, i * self.n_hidden:(i + 1) * self.n_hidden]
                i += 1
            elif name == 'output_layer':
                out = module(out)
                # print(out.shape, n[:, i * self.n_hidden:].shape)
                out += sigma * n[:, i * self.n_hidden:]
            else:
                out = module(out)
        return out


class MultiplicativeNoisyNet(nn.Module):
    pass
