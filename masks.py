import torch
import torch.nn as nn

from torch.distributions.exponential import Exponential
from torch import sigmoid

from collections import OrderedDict


class DifferentiableBernoulli:
    def __init__(self, probs, tau=1):
        """
        :param probs: tensor of success probabilities, can be a parameter
        :param tau: sigmoid temperature
        """
        self.probs = probs
        self.tau = tau
        self.e = Exponential(1)

    def sample_logistic(self, shape):
        X = self.e.sample(shape)
        return torch.log(torch.exp(X) - 1)

    def soft_sample(self, n=1):
        if n == 1:
            shape = self.probs.shape
        else:
            shape = torch.Size([n, *self.probs.shape])
        logistic_sample = self.sample_logistic(shape)
        return sigmoid(1. / self.tau * (logistic_sample + torch.log(self.probs / (1 - self.probs))))

    def sample(self, n=1):
        soft_samples = self.soft_sample(n)
        hard_samples = (soft_samples > 0.5).to(torch.double)
        return hard_samples - soft_samples.detach() + soft_samples


class CustomDropout(nn.Module):
    def __init__(self, premask, tau=1):
        super().__init__()
        # premask should be a 1 dimensional tensor of the inverse sigmoid of success probabilities
        self.premask = premask
        self.tau = tau

    def eval_mask_from_premask(self, premask_input):
        if isinstance(self.premask, nn.Module):
            premask = self.premask(premask_input)
        else:
            premask = self.premask
        db = DifferentiableBernoulli(torch.sigmoid(premask), self.tau)
        return db.sample()

    def forward(self, x, premask_input=None):
        if self.training:
            return x
        else:
            mask = self.eval_mask_from_premask(premask_input)
            # print(mask.shape, x.shape)
            # assert mask.shape[0] == x.shape[1]
            return mask * x


class ModelWithFixedDropout(nn.Module):
    def __init__(self, input_dim=1, n_hidden=32, hidden_layers=1, output_dim=1, tau=1):
        super().__init__()
        self.tau = tau
        activation_fn = nn.ReLU
        self.premask = nn.Parameter(torch.randn(n_hidden, hidden_layers + 1).to(torch.double))
        # premask here can take negative or positive values -> sigmoiding it gives us the success probabilities
        model = nn.Sequential(OrderedDict([
            ('input_layer', nn.Linear(input_dim, n_hidden)),
            ('activation1', activation_fn()),
        ]))
        model.add_module('dropout1', CustomDropout(self.premask[:, 0], tau))
        for i in range(hidden_layers):
            model.add_module('hidden_layer{}'.format(i + 1), nn.Linear(n_hidden, n_hidden))
            model.add_module('activation{}'.format(i + 2), activation_fn())
            model.add_module('dropout{}'.format(i + 2), CustomDropout(self.premask[:, i+1], tau))
        model.add_module('output_layer', nn.Linear(n_hidden, output_dim))
        self.model = model

    def forward(self, x, drop=False):
        return self.model.forward(x)


class ModelWithDropout(nn.Module):
    def __init__(self, input_dim=1, n_hidden=32, hidden_layers=1, output_dim=1, tau=1):
        super().__init__()
        self.tau = tau
        activation_fn = nn.ReLU
        # self.premask_networks = nn.ModuleList([nn.Linear(input_dim, n_hidden) for _ in range(hidden_layers + 1)])
        self.premask_networks = nn.ModuleList([nn.Sequential(OrderedDict([
            ('first', nn.Linear(input_dim, 2 * n_hidden)),
            ('activation', activation_fn()),
            ('second', nn.Linear(2 * n_hidden, 2 * n_hidden)),
            ('second_activation', activation_fn()),
            ('out', nn.Linear(2 * n_hidden, n_hidden))
                                                            ]))
                                               for _ in range(hidden_layers + 1)])
        # nn.Parameter(torch.randn(n_hidden, hidden_layers + 1).to(torch.double))
        model = nn.Sequential(OrderedDict([
            ('input_layer', nn.Linear(input_dim, n_hidden)),
            ('activation1', activation_fn()),
        ]))
        model.add_module('dropout1', CustomDropout(self.premask_networks[0], tau))
        for i in range(hidden_layers):
            model.add_module('hidden_layer{}'.format(i + 1), nn.Linear(n_hidden, n_hidden))
            model.add_module('activation{}'.format(i + 2), activation_fn())
            model.add_module('dropout{}'.format(i + 2), CustomDropout(self.premask_networks[i+1], tau))
        model.add_module('output_layer', nn.Linear(n_hidden, output_dim))
        self.model = model

    def forward(self, x):
        out = x
        for module in self.model:
            if isinstance(module, CustomDropout):
                out = module(out, x)
            else:
                out = module(out)
        return out
