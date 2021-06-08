import torch
import torch.nn as nn
import numpy as np



class Trainer:
    def __init__(self, model, loader, deup_loader, optimizer, deup_optimizer):
        """
        :param model: Instance of networks.NoisyNetwork
        :param loader: DataLoader for the training data
        :param deup_loader: DataLoader for the OOS/OOD data
        :param optimizer: optimizer to train the main regressor
        :param deup_optimizer: DEUP optimizer
        """
        self.model = model
        self.loader = loader
        self.deup_loader = deup_loader
        self.optimizer = optimizer
        self.deup_optimizer = deup_optimizer

        self.train_losses = []
        self.deup_train_losses = []

        self.loss_fn = nn.MSELoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_main(self, epochs):
        for epoch in range(epochs):
            count = 0.
            epoch_losses = []
            for batch_id, (xi, yi) in enumerate(self.loader):
                count += len(xi)
                self.optimizer.zero_grad()

                z = torch.zeros(len(xi), self.model.noise_dim).to(self.device)
                y_hat = self.model(xi, z)

                f_loss = self.loss_fn(y_hat, yi)
                epoch_losses.append(f_loss.item() * xi.shape[0])
                f_loss.backward()
                self.optimizer.step()
            self.train_losses.append(np.sum(epoch_losses) / count)

    def loss_2(self, xi, yi, K=5, z=None):
        # xi and yi are B x ... tensors, with B >=2
        # This returns L2(x, x', y, y', z_1, ..., z_K) for every x, x' in xi, y, y' in yi
        # i.e. it returns a B x B tensor of such losses
        if z is None:
            z = torch.randn(K, self.model.noise_dim).repeat(len(xi), 1, 1).transpose(0, 1).to(self.device)  # K x B x noise_dim
        y_hat = self.model(xi)

        M = self.model(xi, z)  # K x B x 1  ( 1 because its univariate regression)
        M = M.squeeze().transpose(0, 1)  # B x K (M_(i, k)) is f(xi, zk) as a sclaar (univeriate regression)
        predicted_covariance = (1. / (K - 1) * torch.matmul(M, M.T) +
                                1. / (K * (K - 1)) * torch.outer(M.sum(1), M.sum(1).T))  # B x B

        residuals = y_hat - yi  # B x 1
        coresiduals = torch.matmul(residuals, residuals.T)

        return (predicted_covariance - coresiduals).pow(2)

    def loss_1(self, xi, yi, K=5, z=None, alpha_1=1, alpha_2=1):
        # returns a one-dimensional tensor of size B representing L1(x, y)
        # use the same z ? how ? same function? evaluating M twice is redundant
        if z is None:
            z = torch.randn(K, self.model.noise_dim).repeat(len(xi), 1, 1).transpose(0, 1).to(self.device)  # K x B x noise_dim
        print(z.shape)
        M = self.model(xi, z)  # K x B x 1
        y_hat = self.model(xi)
        residuals_squared = (y_hat - yi).pow(2)  # B x 1
        loss_11 = (M.mean(0) - y_hat).pow(2).squeeze()
        loss_12 = (M.var(0) - residuals_squared).pow(2).squeeze()

        return alpha_1 * loss_11 + alpha_2 * loss_12

    def train_deup(self, epochs, K=5, alpha_1=1, alpha_2=1, beta=1):
        for epoch in range(epochs):
            count = 0.
            epoch_losses = []
            for batch_id, (xi, yi) in enumerate(self.deup_loader):
                count += len(xi)
                self.deup_optimizer.zero_grad()

                loss_1 = self.loss_1(xi, yi, K, alpha_1=alpha_1, alpha_2=alpha_2)
                loss_2 = self.loss_2(xi, yi, K)

                loss_1 = loss_1.repeat(len(xi), 1)
                loss_1 = loss_1 + loss_1.T

                loss = loss_1 + beta * loss_2  # B x B matrix
                loss = loss.mean()

                epoch_losses.append(loss.item() * xi.shape[0])
                loss.backward()
                self.deup_optimizer.step()
            self.deup_train_losses.append(np.sum(epoch_losses) / count)



