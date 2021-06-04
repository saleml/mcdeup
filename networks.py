import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.functional as F


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

