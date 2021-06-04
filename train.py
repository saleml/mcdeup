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

                z = torch.randn(len(xi), self.model.noise_dim).to(self.device)
                y_hat = self.model(xi, z)

                f_loss = self.loss_fn(y_hat, yi)
                epoch_losses.append(f_loss.item() * xi.shape[0])
                f_loss.backward()
                self.optimizer.step()
            self.train_losses.append(np.sum(epoch_losses) / count)

    def train_deup(self, epochs):
        alpha = 1
        gamma = 1
        beta = 1
        loss_pow = 1
        for epoch in range(epochs):
            count = 0.
            epoch_losses = []
            for batch_id, (xi, yi) in enumerate(self.deup_loader):
                count += len(xi)
                self.deup_optimizer.zero_grad()

                z = torch.randn(len(xi), self.model.noise_dim).to(self.device)
                y_hat = self.model(xi, z)

                z_prime = torch.randn(len(xi) // 2, self.model.noise_dim).to(self.device)
                xi_1, xi_2 = xi[:len(xi) // 2], xi[len(xi) // 2:]
                yi_1, yi_2 = yi[:len(yi) // 2], yi[len(yi) // 2:]
                mu_1 = self.model(xi_1)
                mu_2 = self.model(xi_2)
                r_hat_1 = self.model(xi_1, z_prime) - mu_1
                r_hat_2 = self.model(xi_2, z_prime) - mu_2
                r_1 = yi_1 - mu_1
                r_2 = yi_2 - mu_2

                f_loss = alpha * self.loss_fn(y_hat, yi) + \
                         gamma * self.loss_fn((r_hat_1 * r_hat_2) ** loss_pow, (r_1 * r_2) ** loss_pow) + \
                         beta * self.loss_fn(r_hat_1 ** loss_pow, r_1 ** loss_pow) + \
                         beta * self.loss_fn(r_hat_2 ** loss_pow, r_2 ** loss_pow)
                epoch_losses.append(f_loss.item() * xi.shape[0])
                f_loss.backward()
                self.deup_optimizer.step()
            self.deup_train_losses.append(np.sum(epoch_losses) / count)



