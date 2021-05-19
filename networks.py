import torch.nn as nn

from old.dropout import (Dropout,
                         RegularDropout, RegularDropoutPerLayer,
                         MultiplicativeGaussianPerLayer)
from noise_generator import GaussianNoiseGenerator
from noise_transformer import ConstantNoiseTransformer


class DropoutModel(nn.Module):
    def __init__(self, input_dim=1, n_hidden=32, hidden_layers=1, output_dim=1, dropout_module=RegularDropout, **kwargs):
        super().__init__()

        activation_fn = nn.ReLU
        model = nn.Sequential()
        for i in range(hidden_layers):
            model.add_module('hidden_layer{}'.format(i + 1), nn.Linear(input_dim if i == 0 else n_hidden, n_hidden))
            model.add_module('activation{}'.format(i + 1), activation_fn())
            if dropout_module in (RegularDropoutPerLayer, MultiplicativeGaussianPerLayer):
                kwargs.update({'layer': i})
            if dropout_module is not None:
                model.add_module('dropout{}'.format(i + 1), dropout_module(**kwargs))
        model.add_module('output_layer', nn.Linear(n_hidden, output_dim))
        self.model = model

    def forward(self, x, **kwargs):
        out = x
        for module in self.model:
            if isinstance(module, Dropout):
                out = module(out, **kwargs)
            else:
                out = module(out)
        return out


class AdditiveNoisyNet(nn.Module):
    def __init__(self, input_dim=1, n_hidden=32, hidden_layers=1,
                 output_dim=1, activation_fn=nn.ReLU, noise_generator=None,
                 noise_transformer=None):
        super().__init__()

        model = nn.Sequential()
        if noise_generator is None:
            noise_generator = GaussianNoiseGenerator(1)
        self.noise_generator = noise_generator
        if noise_transformer is None:
            noise_transformer = ConstantNoiseTransformer(output_dim + n_hidden * (hidden_layers - 1))
        self.noise_transformer = noise_transformer

        self.n_hidden = n_hidden

        model.add_module('input_layer', nn.Linear(input_dim, n_hidden))
        model.add_module('activation0', activation_fn())
        for i in range(hidden_layers - 1):
            model.add_module('hidden_layer{}'.format(i + 1), nn.Linear(n_hidden, n_hidden))
            model.add_module('activation{}'.format(i + 1), activation_fn())
        model.add_module('output_layer', nn.Linear(n_hidden, output_dim))
        self.model = model

    def forward(self, x, sigma=1.):
        z = self.noise_generator.sample(x)
        n = self.noise_transformer(z)  # should be of shape B x n_units
        out = x
        i = 0
        for (name, module) in self.model.named_modules():
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