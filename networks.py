import torch.nn as nn

from dropout import (Dropout,
                     RegularDropout, RegularDropoutPerLayer,
                     MultiplicativeGaussian, MultiplicativeGaussianPerLayer)


class Model(nn.Module):
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
