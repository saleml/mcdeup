import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np


class Trainer:
    """
    Class to handle all training related code: training main model, training uncertainty estimators, etc...
    model has to be a networks.NoisyNet instance
    """
    def __init__(self, model, optimizer, train_dataset, validation_dataset,
                 batch_size, dataloader_seed=0, loss_fn=nn.MSELoss(), deup_optimizer=None,
                 deup_dataset=None):
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.validation_dataset = validation_dataset
        self.train_dataset = train_dataset
        self.deup_dataset = deup_dataset

        self.dataloader_seed = dataloader_seed
        self.train_loader = self.make_loader(self.train_dataset)
        if deup_dataset is not None:
            self.deup_loader = self.make_loader(self.deup_dataset)

        self.model = model
        self.optimizer = optimizer
        self.deup_optimizer = deup_optimizer

        self.train_losses = []
        self.valid_losses = []
        self.deup_losses = []

    def make_loader(self, dataset):
        generator = torch.Generator().manual_seed(self.dataloader_seed)
        sampler = RandomSampler(dataset, generator=generator)
        loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        return loader

    def train(self, epochs, sigma=0., fixed_noise=False):
        for epoch in range(epochs):
            self.train_epoch(sigma, fixed_noise)

    def train_epoch(self, sigma, fixed_noise):
        epoch_losses = []
        for batch_id, (xi, yi) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            y_hat = self.model(xi, sigma, fixed_noise)
            loss = self.loss_fn(y_hat, yi)
            epoch_losses.append(loss.item() * xi.shape[0])
            loss.backward()
            self.optimizer.step()
        self.train_losses.append(np.sum(epoch_losses) / self.train_dataset[:][0].shape[0])
        self.valid_losses.append(self.loss_fn(self.model(self.validation_dataset[:][0], sigma, fixed_noise),
                                              self.validation_dataset[:][1]))

    def train_deup(self, epochs, K=2):
        for epoch in range(epochs):
            self.train_deup_epoch(K)

    def train_deup_epoch(self, K=2):
        epoch_losses = []
        for batch_id, (xi, yi) in enumerate(self.deup_loader):
            self.deup_optimizer.zero_grad()
            corrupted_outputs = torch.stack([self.model(xi, sigma=1.) for _ in range(K)])  # K x B x out tensor
            uncorrupted_output = self.model(xi, sigma=0.)
            differences = (corrupted_outputs - uncorrupted_output).pow(2)

            residual = (yi - uncorrupted_output).pow(2)
            loss = self.loss_fn(differences.mean(0), residual)

            sample_variances = (K + 1) / (K - 1) * differences.var(0)  # Should be B x out
            loss -= sample_variances.mean(0).mean()

            epoch_losses.append(loss.item() * xi.shape[0])
            if loss.item() > 1e3:
                print(sample_variances.mean(0).mean().item(), self.loss_fn(differences.mean(0), residual).item())
            loss.backward()
            self.deup_optimizer.step()
        self.deup_losses.append(np.sum(epoch_losses) / self.train_dataset[:][0].shape[0])


