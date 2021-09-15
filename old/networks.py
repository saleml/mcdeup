import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
import numpy as np
import torch


class NoisyNetwork(nn.Module, ABC):
    """
    Base class for all distribution inducing networks
    """

    @abstractmethod
    def forward(self, x, z=None):
        """
        :param x: main input of the network of size B x ...
        :param z: noise vector of size B x r where r is the dimensionality of the noise used
        :return: f(x, z)
        """


class main_net(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.x_layer = nn.Linear(1, width)
        self.hidden_layer_1 = nn.Linear(width, width)
        self.hidden_layer_2 = nn.Linear(width, width)

    def forward(self, x, noise_1=None, noise_2=None):
        out = F.relu(self.x_layer(x))
        if noise_1 is not None:
            out = F.relu(self.hidden_layer_1(out)) * noise_1
        else:
            out = F.relu(self.hidden_layer_1(out))
        return self.hidden_layer_2(out) * noise_2 if noise_2 is not None else self.hidden_layer_2(out)


output_net = nn.Sequential(nn.Linear(256, 128),
                           nn.ReLU(),
                           nn.Linear(128, 1))


class MultiplicativeNetwork(NoisyNetwork):
    class NoiseNet(nn.Module):
        def __init__(self, noise_dim=64, width=256, output_dim=256):
            super().__init__()
            self.noise_layer = nn.Linear(noise_dim, width, bias=False)
            # self.x_layer = nn.Linear(1, width, bias=True)
            self.hidden_layer_1 = nn.Linear(width, width, bias=True)
            self.hidden_layer_2 = nn.Linear(width, width, bias=False)
            self.output_layer = nn.Linear(width, output_dim, bias=False)

        def forward(self, x, z):
            if z is None:
                return 1
            # the x input is if we want this noisy part of the network to have x as input as well
            # out = F.relu(self.x_layer(x))
            # out = F.relu(self.hidden_layer_1(out)) * self.noise_layer(noise)
            out = F.relu(self.noise_layer(z))
            out = F.relu(self.hidden_layer_1(out))
            out = F.relu(self.hidden_layer_2(out))
            return 1 + self.output_layer(out)

    def __init__(self, input_dim=1, output_dim=1, width=256, noise_dim=64):
        super().__init__()
        self.x_layer = nn.Linear(input_dim, width)
        self.hidden_layer_1 = nn.Linear(width, width)
        self.hidden_layer_2 = nn.Linear(width, width)
        self.noise_net_1 = self.NoiseNet(noise_dim, width=width, output_dim=width)
        self.noise_net_2 = self.NoiseNet(noise_dim, width=width, output_dim=width)
        self.output_net = nn.Sequential(nn.Linear(width, width // 2),
                                        nn.ReLU(),
                                        nn.Linear(width // 2, output_dim))
        self.noise_dim = noise_dim

    def forward(self, x, z=None):
        noise_1 = self.noise_net_1(x, z)
        noise_2 = self.noise_net_2(x, z)
        out = F.relu(self.x_layer(x))
        out = F.relu(self.hidden_layer_1(out)) * noise_1
        out = self.hidden_layer_2(out) * noise_2
        out = self.output_net(out)
        return out


class OnePlusId(nn.Module):
    def forward(self, x):
        return 1. + x


class NoiseNet(nn.Module):
    def __init__(self, noise_dim=64, depth=2, width=256, output_dim=256, use_x=False, x_dim=1, final_layer=None):
        super().__init__()
        self.depth = depth
        self.use_x = use_x
        if use_x:
            self.x_layer = nn.Linear(x_dim, width // 2)
        self.noise_layer = nn.Linear(noise_dim,
                                     width - width // 2 if use_x else width,
                                     bias=False)  # why no bias
        # self.x_layer = nn.Linear(1, width, bias=True)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.output_layer = nn.Linear(width, output_dim, bias=False)  # why no bias ?
        if final_layer is None:
            self.final_layer = OnePlusId()
        elif final_layer == 'softplus':
            self.final_layer = nn.Softplus(beta=np.log(2))
        else:
            raise NotImplementedError('No final layer with this name')

    def forward(self, x, z):
        if z is None:
            return 1
        # the x input is if we want this noisy part of the network to have x as input as well
        # out = F.relu(self.x_layer(x))
        # out = F.relu(self.hidden_layer_1(out)) * self.noise_layer(noise)
        out = F.relu(self.noise_layer(z))
        if self.use_x:
            x_hid = F.relu(self.x_layer(x))
            if x_hid.ndim == 2 and z.ndim == 3:
                x_hid = x_hid.repeat(z.shape[0], 1, 1)
            out = torch.cat([out, x_hid], -1)
        for i in range(self.depth):
            out = F.relu(self.hidden_layers[i](out))
        out = self.output_layer(out)
        return self.final_layer(out)


class MultiplicativeNetworkTmp(NoisyNetwork):
    def __init__(self, input_dim=1, output_dim=1, depth=2, width=256, noise_dim=64, noise_depth=2, noise_width=256,
                 bottleneck_ratio=2, use_x=False, noise_final_layer=None):
        super().__init__()
        self.depth = depth
        self.x_layer = nn.Linear(input_dim, width)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.noise_nets = nn.ModuleList([NoiseNet(noise_dim, depth=noise_depth,
                                                  width=noise_width, output_dim=width,
                                                  final_layer=noise_final_layer, use_x=use_x) for _ in range(depth)])
        self.output_net = nn.Sequential(nn.Linear(width, width // bottleneck_ratio),
                                        nn.ReLU(),
                                        nn.Linear(width // bottleneck_ratio, output_dim))
        self.noise_dim = noise_dim

    def forward(self, x, z=None):
        noises = [self.noise_nets[i](x, z) for i in range(self.depth)]
        out = F.relu(self.x_layer(x))
        for i in range(self.depth):
            out = F.relu(self.hidden_layers[i](out)) * noises[i]
        # out = self.hidden_layers[-1](out) * noises[-1]  # why no relu here ?
        out = self.output_net(out)
        return out


