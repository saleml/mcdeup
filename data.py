import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)





def generate_data(n=128, n_oos=24, n_ood=24, device=torch.device('cpu'), seed=1,
                  plot=False, one_sine=False, distance=1.5, shuffle=False, offset=0.):
    torch.manual_seed(seed)
    
    def oracle(x, noise=0):
        # Sinusoid function, highly perturbed between 0.5 and 1.5
        with torch.no_grad():
            m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1]))
            m = m.sample((x.size(0),))
        return torch.where(torch.logical_and(x > 0.5, x < distance),
                           3./5 * (torch.sin(x) * torch.cos(5 * x) * torch.cos(22 * x) * -3 * torch.sin(7 * x) * torch.cos(19 * x) * 4 * torch.sin(11 * x)),
               (2 * np.pi * x).sin()) + noise * m
    
    def split(x_data, n_data):
        y_data = oracle(x_data).to(device)
        split1, split2 = random_split(TensorDataset(x_data, y_data), (n_data // 2, n_data - n_data // 2))
        return split1[:], split2[:]

    with torch.no_grad():
        x = torch.zeros(n // 3, 1).uniform_(0, 0.5).to(device)
        x = torch.cat((x, torch.zeros(n // 6, 1).uniform_(-0.5, 0.))).to(device)
        if not one_sine:
            x = torch.cat((x, torch.zeros(n // 3, 1).uniform_(distance, distance+0.5)), 0).to(device)
            x = torch.cat((x, torch.zeros(n // 6, 1).uniform_(distance+0.5, distance+1.))).to(device)
        y = oracle(x).to(device)
        
        if shuffle:
            ind = torch.randperm(x.size(0))
            x = x[ind, :]
            y = y[ind, :]

        oos_x = torch.zeros(n_oos // 2, 1).uniform_(-0.5, 0.0).to(device)
        oos_x = torch.cat((oos_x, torch.zeros(n_oos // 2, 1).uniform_(distance+0.5, distance+1.)), 0).to(device)
        if shuffle:
            ind = torch.randperm(oos_x.size(0))
            oos_x = oos_x[ind, :]
        (oos_x, oos_y), (oos_x_test, oos_y_test) = split(oos_x, n_oos)

        ood_x = torch.zeros(n_ood, 1).uniform_(0.5, distance).to(device)
        if shuffle:
            ind = torch.randperm(ood_x.size(0))
            ood_x = ood_x[ind, :]
        (ood_x, ood_y), (ood_x_test, ood_y_test) = split(ood_x, n_ood)
        
        x_test = torch.linspace(-0.5, distance+1, 1024).view(-1, 1).to(device)
        y_test = oracle(x_test, noise=0).to(device)

    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(x+offset, y.squeeze(), 'ro', label='training data')
        plt.plot(ood_x+offset, ood_y.squeeze(), 'b.', label='OOD data')
        plt.plot(ood_x_test+offset, ood_y_test.squeeze(), 'bx', label='testOOD data')
        plt.plot(oos_x+offset, oos_y.squeeze(), 'g.', label='OOS data')
        plt.plot(oos_x_test+offset, oos_y_test.squeeze(), 'gx', label='testOOS data')
        plt.plot(x_test+offset, y_test.squeeze(), label='ground truth function', linewidth=.3)
        plt.legend()
        plt.show()

    return x+offset, y, oos_x+offset, oos_y, ood_x+offset, ood_y, oos_x_test+offset, oos_y_test, ood_x_test+offset, ood_y_test, x_test+offset, y_test

