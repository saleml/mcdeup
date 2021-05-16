import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def train_main_model(model, optimizer, train_data, validation_data, epochs=500, batch_size=32):
    loss_fn = nn.MSELoss()

    model.train()
    loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    losses = []
    valid_losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch_id, (xi, yi) in enumerate(loader):
            optimizer.zero_grad()

            y_hat = model(xi)
            f_loss = loss_fn(y_hat, yi)
            epoch_losses.append(f_loss.item() * xi.shape[0])
            f_loss.backward()
            optimizer.step()
        losses.append(np.sum(epoch_losses) / train_data[:][0].shape[0])
        valid_losses.append(loss_fn(model(validation_data[:][0]), validation_data[:][1]))
    return losses, valid_losses


def train_deup(model, optimizer, full_data, epochs=500, batch_size=32, n_samples=5, parametric_noise=False, x_input=False, callback=None):
    loss_fn = nn.MSELoss()

    loader = DataLoader(full_data, shuffle=True, batch_size=batch_size)
    losses = []
    callbacks = []

    for epoch in range(epochs):
        if callback is not None:
            callbacks.append(callback(model))
        epoch_losses = []
        for batch_id, (xi, yi) in enumerate(loader):
            optimizer.zero_grad()
            model.eval()
            y_mean = model(xi)
            model.train()
            y_noisy = torch.cat([model(xi, parametric_noise=parametric_noise, noise_input=xi if x_input else None)
                                 for _ in range(n_samples)], 1)
            pred_uncertainties = ((y_mean - y_noisy) ** 2)
            pred_uncertainty = pred_uncertainties.mean(1).unsqueeze(-1)
            variance_estimator = (n_samples / n_samples - 1) * torch.var(pred_uncertainties, 1)

            true_uncertainty = (yi - y_mean) ** 2
            f_loss = loss_fn(pred_uncertainty, true_uncertainty) - (n_samples + 1) / n_samples * variance_estimator.mean()

            epoch_losses.append(f_loss.item() * xi.shape[0])
            f_loss.backward()
            optimizer.step()

        losses.append(np.sum(epoch_losses) / full_data[:][0].shape[0])
    return losses, callbacks


def train_regular_deup(model, deup_model, optimizer, full_data, epochs=500, batch_size=32, callback=None):
    loss_fn = nn.MSELoss()

    loader = DataLoader(full_data, shuffle=True, batch_size=batch_size)
    losses = []
    callbacks = []

    for epoch in range(epochs):
        if callback is not None:
            callbacks.append(callback(model))
        epoch_losses = []
        for batch_id, (xi, yi) in enumerate(loader):
            optimizer.zero_grad()
            model.eval()
            y_mean = model(xi)
            true_uncertainty = (yi - y_mean) ** 2

            pred_uncertainty = deup_model(xi).pow(2).sum(1, keepdims=True)  # deup_model outputs N x d tensors, this transforms it to N x 1
            deup_model.train()

            f_loss = loss_fn(pred_uncertainty, true_uncertainty)

            epoch_losses.append(f_loss.item() * xi.shape[0])
            f_loss.backward()
            optimizer.step()

        losses.append(np.sum(epoch_losses) / full_data[:][0].shape[0])
    return losses, callbacks


def train_regular_deup_plusplus(model, deup_model, optimizer, full_data, epochs=500, batch_size=32, callback=None):
    loss_fn = nn.MSELoss()
    loader = DataLoader(full_data, shuffle=True, batch_size=batch_size)
    losses = []
    callbacks = []

    for epoch in range(epochs):
        if callback is not None:
            callbacks.append(callback(model))
        epoch_losses = []
        for batch_id, (xi, yi) in enumerate(loader):
            optimizer.zero_grad()
            model.eval()
            y_mean = model(xi)
            differences = (yi - y_mean).squeeze()  # should be a 1D array
            empirical_covariance = torch.outer(differences, differences)
            pred_differences = deup_model(xi).squeeze()  # should be a 1D array
            pred_covariance = torch.outer(pred_differences, pred_differences)

            # f_loss = (empirical_covariance - pred_covariance).pow(2).diag().mean()
            f_loss = (empirical_covariance - pred_covariance).pow(2).mean()
            epoch_losses.append(f_loss.item() * xi.shape[0])
            f_loss.backward()
            optimizer.step()

        losses.append(np.sum(epoch_losses) / full_data[:][0].shape[0])
    return losses, callbacks
