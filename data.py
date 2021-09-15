import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)


def oracle(x, dist, noise=0):
    # Sinusoid function, highly perturbed between -0.5 and 0.5
    with torch.no_grad():
        m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1]))
        m = m.sample((x.size(0),))
    return torch.where(torch.logical_and(x > -0.5, x < dist),
                       3. / 5 * (torch.sin(x) * torch.cos(5 * x) * torch.cos(22 * x) * -3 * torch.sin(
                           7 * x) * torch.cos(19 * x) * 4 * torch.sin(11 * x)),
                       (2 * np.pi * x).sin()) + noise * m


def linear_quadratic(x, dist, a=1, noise=0):
    with torch.no_grad():
        m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1]))
        m = m.sample((x.size(0),))
    return torch.where(torch.logical_and(x > -0.5, x < dist),
                       0.5 * a * torch.sin(np.pi * x) - 4 * a * x ** 2 + a,
                       a * x)


def generate_data(n=128, n_oos=24, n_ood=24, device=torch.device('cpu'), seed=1,
                  plot=False, one_sine=False, distance=0.5, shuffle=False, offset=0., oracle=oracle):
    torch.manual_seed(seed)

    def split(x_data, n_data):
        y_data = oracle(x_data, dist=distance)
        split1, split2 = random_split(TensorDataset(x_data, y_data), (n_data // 2, n_data - n_data // 2))
        return split1[:], split2[:]

    with torch.no_grad():
        x = torch.zeros(n // 3, 1).uniform_(-1., -0.5)
        x = torch.cat((x, torch.zeros(n // 6, 1).uniform_(-1.5, -1.)))
        if not one_sine:
            x = torch.cat((x, torch.zeros(n // 3, 1).uniform_(distance, distance+0.5)), 0)
            x = torch.cat((x, torch.zeros(n // 6, 1).uniform_(distance+0.5, distance+1.)))
        y = oracle(x, dist=distance)
        
        if shuffle:
            ind = torch.randperm(x.size(0))
            x = x[ind, :]
            y = y[ind, :]

        oos_x = torch.zeros(n_oos // 2, 1).uniform_(-1.5, -1.0)
        oos_x = torch.cat((oos_x, torch.zeros(n_oos // 2, 1).uniform_(distance+0.5, distance+1.)), 0)
        if shuffle:
            ind = torch.randperm(oos_x.size(0))
            oos_x = oos_x[ind, :]
        (oos_x, oos_y), (oos_x_test, oos_y_test) = split(oos_x, n_oos)

        ood_x = torch.zeros(n_ood, 1).uniform_(-0.5, distance)
        if shuffle:
            ind = torch.randperm(ood_x.size(0))
            ood_x = ood_x[ind, :]
        (ood_x, ood_y), (ood_x_test, ood_y_test) = split(ood_x, n_ood)
        
        x_test = torch.linspace(-1.5, distance+1, 1024).view(-1, 1)
        y_test = oracle(x_test, dist=distance, noise=0)

    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(x+offset, y.squeeze(), 'ro', label='training data')
        plt.plot(ood_x+offset, ood_y.squeeze(), 'b.', label='OOD data')
        # plt.plot(ood_x_test+offset, ood_y_test.squeeze(), 'bx', label='testOOD data')
        plt.plot(oos_x+offset, oos_y.squeeze(), 'g.', label='OOS data')
        # plt.plot(oos_x_test+offset, oos_y_test.squeeze(), 'gx', label='testOOS data')
        plt.plot(x_test+offset, y_test.squeeze(), label='ground truth function', linewidth=.3)
        plt.legend()
        plt.show()

    return (x+offset).to(device), y.to(device), \
           (oos_x+offset).to(device), oos_y.to(device), \
           (ood_x+offset).to(device), ood_y.to(device), \
           (oos_x_test+offset).to(device), oos_y_test.to(device), \
           (ood_x_test+offset).to(device), ood_y_test.to(device), \
           (x_test+offset).to(device), y_test.to(device)

