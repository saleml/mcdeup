import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class NoiseTransformer(nn.Module, ABC):
    """
    Base class for the different noise transformers
    """
    def __init__(self, n_units):
        """
        :param n_units: int, total nomber of neural network units to perturb
        """
        super().__init__()
        self.n_units = n_units

    @abstractmethod
    def forward(self, z):
        """
        :param z: tensor, of shape B x z_in, source vectors of independent noises
        :return: tensor of shape B x n_units, noises to be added to each unit
        """
        pass


class ConstantNoiseTransformer(NoiseTransformer):
    """
    Same noise value for each unit
    """
    def forward(self, z):
        return z[:, 0].unsqueeze(-1).repeat(1, self.n_units)


class BlockNoiseTransformer(NoiseTransformer):
    """
    Blocks of units with similar value of noise
    """
    def __init__(self, n_units, B=None):
        # If B is None, then B = n_units, and we obtain ConstantNoiseTransformer
        super().__init__(n_units)
        if B is None:
            B = self.n_units
        self.B = B

    def forward(self, z):
        z_repeat = torch.repeat_interleave(z, self.B, 1).repeat(1, self.n_units // self.B)
        return z_repeat[:, :self.n_units]


class NNNoiseTransformer(NoiseTransformer):
    """
    Neural net based noise transformer. forward takes an extra input (x)
    """
    pass

