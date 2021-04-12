import torch
import torch.nn as nn

from collections import OrderedDict
from dropout import FixedDropout, FixedMultiplicativeGaussian, FixedMultiplicativeGaussianPerLayer, Dropout


class ModelWithFixedDropout(nn.Module):
    def __init__(self, input_dim=1, n_hidden=32, hidden_layers=1, output_dim=1, dropout_module=FixedDropout, **kwargs):
        super().__init__()
        assert dropout_module in (FixedDropout, FixedMultiplicativeGaussian, FixedMultiplicativeGaussianPerLayer),\
            dropout_module

        activation_fn = nn.ReLU
        model = nn.Sequential(OrderedDict([
            ('input_layer', nn.Linear(input_dim, n_hidden)),
            ('activation1', activation_fn()),
        ]))
        if dropout_module == FixedMultiplicativeGaussianPerLayer:
            kwargs.update({'layer': 0})
        model.add_module('dropout1', dropout_module(**kwargs))
        for i in range(hidden_layers):
            model.add_module('hidden_layer{}'.format(i + 1), nn.Linear(n_hidden, n_hidden))
            model.add_module('activation{}'.format(i + 2), activation_fn())
            if dropout_module == FixedMultiplicativeGaussianPerLayer:
                kwargs.update({'layer': i + 1})
            model.add_module('dropout{}'.format(i + 2), dropout_module(**kwargs))
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
