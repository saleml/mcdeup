import torch
import matplotlib.pyplot as plt


def get_dist(model, x, num_samples=10):
    model.train()
    outputs = torch.cat([model(x).unsqueeze(0) for _ in range(num_samples)])
    y_mean = outputs.mean(0)
    y_std = outputs.std(0)
    return y_mean.detach().squeeze(), y_std.detach().squeeze()


def eval_error(model_drop, X, Y, std=False):
    model_drop.eval()
    y_mean = model_drop(X)
    model_drop.train()

    if std:
        y_noisy = torch.cat([model_drop(X) for _ in range(10)], 1)
        pred_uncertainty = y_noisy.var(dim=1).squeeze()
    else:
        y_noisy = model_drop(X)
        pred_uncertainty = (y_mean - y_noisy) ** 2
    true_uncertainty = (Y - y_mean) ** 2
    return ((pred_uncertainty - true_uncertainty) ** 2).mean()


def get_dist_deup(model_drop, x):
    model_drop.eval()
    y_mean = model_drop(x)
    model_drop.train()
    y_noisy = model_drop(x)
    y_std = torch.abs(y_mean - y_noisy)
    return y_mean.detach().squeeze(), y_std.detach().squeeze()


def evaluate_and_plot(model, x_test, full_X, full_Y, oos_ood=None, test_data=None, deup=False):
    model.eval()
    non_dropout_pred = model(x_test).detach()
    if deup:
        Y_mean, Y_std = get_dist_deup(model, x_test)
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
