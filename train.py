import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def train_main_model(model, train_data, validation_data, epochs=500, lr=1e-3, batch_size=32):
    optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if 'dropout' not in name],
                                 lr=lr)
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