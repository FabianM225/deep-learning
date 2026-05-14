import numpy as np
import scipy.sparse.linalg
import torch
from torch import nn
from typing import Tuple
import graphtools as gt
import networkx as nx

from utils import calcMI_torch


def get_laplacian_extrema(data, n_extrema, knn=10, subsample=True):
    if subsample and data.shape[0] > 10000:
        data = data[np.random.choice(data.shape[0], 10000, replace=False), :]

    G = gt.Graph(data, use_pygsp=True, decay=None, knn=knn)
    G_nx = nx.convert_matrix.from_scipy_sparse_array(G.W)

    fiedler = nx.linalg.algebraicconnectivity.fiedler_vector(
        G_nx, method='tracemin_pcg'
    )

    L = nx.laplacian_matrix(G_nx)

    first_extrema = np.argmax(fiedler)
    extrema = [first_extrema]
    init_lanczos = fiedler.copy()

    for _ in range(n_extrema - 1):
        remaining = np.setdiff1d(np.arange(data.shape[0]), extrema)
        L_sub = L[remaining][:, remaining]

        eigvals, eigvecs = scipy.sparse.linalg.eigsh(
            L_sub, k=1, which='SM', v0=init_lanczos[remaining]
        )
        eigvec = eigvecs[:, 0]
        idx_sub = np.argmax(np.abs(eigvec))
        new_extrema = remaining[idx_sub]
        extrema.append(new_extrema)

        init_lanczos = np.zeros_like(fiedler)
        init_lanczos[remaining] = eigvec

    return extrema


def train_epoch(model, data_loader, optimizer, epoch,
                gamma_reconstruction=1.0,
                gamma_archetypal=1.0,
                gamma_extrema=1e-4,
                gamma_mi=0.1):
    loss = 0
    reconstruction_loss = 0
    archetypal_loss = 0
    mi_loss_epoch = 0

    for idx, data in enumerate(data_loader):
        if isinstance(data, list):
            batch_features = data[0]
        else:
            batch_features = data

        if model.diffusion_extrema is not None:
            batch_features = torch.cat((
                model.diffusion_extrema.view(-1, model.input_shape),
                batch_features.view(-1, model.input_shape)
            ), 0)

        batch_features = batch_features.view(-1, model.input_shape)
        batch_features = batch_features.to(model.device).float()

        optimizer.zero_grad()
        output, _in, archetypal_embedding = model(batch_features)

        curr_reconstruction_loss = torch.mean((output - batch_features)**2)
        reconstruction_loss += curr_reconstruction_loss

        curr_archetypal_loss = model.calc_archetypal_loss(archetypal_embedding)
        archetypal_loss += curr_archetypal_loss

        if model.diffusion_extrema is not None:
            curr_extrema_loss = model.calc_diffusion_extrema_loss(archetypal_embedding)
        else:
            curr_extrema_loss = 0

        if model.diffusion_extrema is not None:
            z = archetypal_embedding[len(model.diffusion_extrema):]
        else:
            z = archetypal_embedding

        z = torch.softmax(z, dim=1)
        curr_mi_loss = calcMI_torch(z, z)
        mi_loss_epoch += curr_mi_loss

        train_loss = (
            gamma_reconstruction * curr_reconstruction_loss +
            gamma_archetypal     * curr_archetypal_loss +
            gamma_extrema / (epoch * len(data_loader) + (idx + 1)) * curr_extrema_loss +
            gamma_mi             * curr_mi_loss
        )

        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    N = len(data_loader)
    return loss / N, reconstruction_loss / N, archetypal_loss / N, mi_loss_epoch / N


class AAnet_vanilla(nn.Module):
    def __init__(
        self,
        input_shape,
        n_archetypes=4,
        noise=0,
        layer_widths=[128, 128],
        activation_out="tanh",
        simplex_scale=1,
        device=None,
        diffusion_extrema=None,
        **kwargs
    ):
        super().__init__()

        self.input_shape = input_shape
        self.n_archetypes = n_archetypes
        self.noise = noise
        self.layer_widths = layer_widths
        self.activation_out = activation_out
        self.simplex_scale = simplex_scale
        self.diffusion_extrema = diffusion_extrema

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.encoder_layers = nn.ModuleList()
        for i, width in enumerate(layer_widths):
            in_features = input_shape if i == 0 else layer_widths[i - 1]
            self.encoder_layers.append(nn.Linear(in_features, width))
        self.encoder_layers.append(nn.Linear(layer_widths[-1], n_archetypes - 1))

        self.decoder_layers = nn.ModuleList()
        decoder_widths = layer_widths[::-1]
        for i, width in enumerate(decoder_widths):
            in_features = n_archetypes - 1 if i == 0 else decoder_widths[i - 1]
            self.decoder_layers.append(nn.Linear(in_features, width))
        self.decoder_layers.append(nn.Linear(decoder_widths[-1], input_shape))

        self.archetypal_simplex = self.get_n_simplex(n_archetypes, simplex_scale)
        self.to(self.device)

    def get_n_simplex(self, n=2, scale=1):
        nth = (1 / (n - 1) * (1 - np.sqrt(n))) * np.ones(n - 1)
        D = np.vstack([np.eye(n - 1), nth]) * scale
        return torch.tensor(D - np.mean(D, axis=0), dtype=torch.float, device=self.device)

    def euclidean_to_barycentric(self, X):
        simplex = self.archetypal_simplex
        T = torch.zeros((X.shape[1], X.shape[1])).to(self.device)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                T[i, j] = simplex[i, j] - simplex[-1, j]
        T_inv = torch.inverse(T).float().to(self.device)
        X_bary = torch.einsum("ij,bj->bi", T_inv, X - simplex[-1])
        X_bary = torch.cat(
            [X_bary, (1 - torch.sum(X_bary, dim=1, keepdim=True))], dim=1
        )
        return X_bary

    def dist_to_simplex(self, X_bary):
        return torch.sum(torch.clamp(-X_bary, min=0), dim=1)

    def calc_archetypal_loss(self, archetypal_embedding):
        X_bary = self.euclidean_to_barycentric(archetypal_embedding)
        return torch.mean(self.dist_to_simplex(X_bary) ** 2)

    def calc_diffusion_extrema_loss(self, archetypal_embedding):
        X_bary = self.euclidean_to_barycentric(archetypal_embedding)
        return torch.mean(
            (X_bary[:self.n_archetypes, :] -
             torch.eye(self.n_archetypes, device=self.device)) ** 2
        )

    def encode(self, x):
        for layer in self.encoder_layers[:-1]:
            x = torch.relu(layer(x))
        return self.encoder_layers[-1](x)

    def decode(self, z):
        for layer in self.decoder_layers[:-1]:
            z = torch.relu(layer(z))
        return self.decoder_layers[-1](z)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        archetypal_embedding = z.clone()
        if self.noise > 0:
            z = z + torch.normal(0., self.noise, size=z.shape).to(self.device)
        recons = self.decode(z)
        return recons, x, archetypal_embedding
