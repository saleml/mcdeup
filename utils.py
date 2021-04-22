import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.exponential import Exponential
from torch import sigmoid


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

    def soft_sample(self, shape=None):
        if shape is None:
            shape = self.probs.shape
        logistic_sample = self.sample_logistic(shape)
        return sigmoid(1. / self.tau * (logistic_sample + torch.log(self.probs / (1 - self.probs))))

    def sample(self, shape, soft=False):
        soft_samples = self.soft_sample(shape)
        if soft:
            return soft_samples
        hard_samples = (soft_samples > 0.5).to(torch.double)
        return hard_samples - soft_samples.detach() + soft_samples


def inv_sigmoid(p):
    return np.log(p / (1 - p))


def get_dist(model, x, num_samples=10):
    model.train()
    outputs = torch.cat([model(x).unsqueeze(0) for _ in range(num_samples)])
    y_mean = outputs.mean(0)
    y_std = outputs.std(0)
    return y_mean.detach().squeeze(), y_std.detach().squeeze()


def eval_error(model, X, Y, n_samples=1, parametric_noise=False, x_input=False):
    model.eval()
    y_mean = model(X)
    model.train()

    y_noisy = torch.cat([model(X, parametric_noise=parametric_noise, noise_input=X if x_input else None)
                                 for _ in range(n_samples)], 1)
    pred_uncertainties = ((y_mean - y_noisy) ** 2)
    pred_uncertainty = pred_uncertainties.mean(1).unsqueeze(-1)

    true_uncertainty = (Y - y_mean) ** 2
    return ((pred_uncertainty - true_uncertainty) ** 2).mean()


def get_dist_deup(model_drop, x, n_samples=1, parametric_noise=False, x_input=False):
    model_drop.eval()
    y_mean = model_drop(x)
    model_drop.train()
    y_noisy = torch.cat([model_drop(x, parametric_noise=parametric_noise, noise_input=x if x_input else None)
                         for _ in range(n_samples)], 1)
    y_std = torch.abs(y_mean - y_noisy).mean(1)
    return y_mean.detach().squeeze(), y_std.detach().squeeze()


def evaluate_and_plot(model, x_test, full_X, full_Y, oos_ood=None, test_data=None, deup=False, parametric_noise=False,
                      x_input=False, n_samples=1):
    plt.figure(figsize=(15, 5))
    model.eval()
    non_dropout_pred = model(x_test).detach()
    if deup:
        Y_mean, Y_std = get_dist_deup(model, x_test, n_samples=n_samples, parametric_noise=parametric_noise, x_input=x_input)
    else:
        Y_mean, Y_std = get_dist(model, x_test, num_samples=100)
    plt.plot(full_X, full_Y, 'r.', label='train_data')
    if oos_ood is not None:
        plt.plot(oos_ood[0], oos_ood[1], 'c.', label='OOD and OOS for deup')
    if test_data is not None:
        plt.plot(test_data[:][0], test_data[:][1], 'cx', label='test_data')
    # plt.scatter(x_test, Y_mean, s=.2, label='mean of dropout masks')

    plt.plot(x_test, non_dropout_pred, 'k', label='model prediction (no unit dropped)')

    plt.fill_between(x_test.squeeze(), Y_mean - Y_std, Y_mean + Y_std, color='crimson', alpha=.2)
    plt.legend()
    plt.show()
