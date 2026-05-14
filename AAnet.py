"""
AAnet (Archetypal Analysis Network) training script.

Usage:
    python AAnet.py --dataset mnist
    python AAnet.py --dataset blood --subset 0.5 --device cuda --epochs 50

Saves: results/aanet_{dataset}_results.pt
"""

import argparse
import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

from aanetOrig import get_laplacian_extrema, train_epoch, AAnet_vanilla
from utils import ArchetypeConsistency, preprocess

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist(subset):
    from torchvision import datasets, transforms
    mnist = datasets.MNIST(root='./data', train=True, download=True,
                           transform=transforms.ToTensor())
    X = mnist.data.numpy().reshape(60000, -1)
    y = mnist.targets.numpy()
    rng = np.random.default_rng(42)
    idx = np.hstack([rng.choice(np.where(y == d)[0],
                                max(1, int(subset * np.sum(y == d))), replace=False)
                     for d in np.unique(y)])
    X_sub = X[idx]
    data = torch.from_numpy(((X_sub / 255.0) * 2 - 1)).float()
    N = len(data)
    perm = torch.randperm(N)
    t, v = int(0.7 * N), int(0.85 * N)
    return data[perm[:t]], data[perm[t:v]], data[perm[v:]], y[idx[perm.numpy()]]


def load_blood(subset):
    from medmnist import BloodMNIST
    blood = BloodMNIST(split='train', download=True, size=28)
    X = blood.imgs.reshape(len(blood.imgs), -1)
    y = blood.labels.squeeze()
    rng = np.random.default_rng(42)
    idx = np.hstack([rng.choice(np.where(y == d)[0],
                                max(1, int(subset * np.sum(y == d))), replace=False)
                     for d in np.unique(y)])
    X_sub = X[idx]
    data = torch.from_numpy(((X_sub.astype(np.float32) / 255.0) * 2 - 1))
    N = len(data)
    perm = torch.randperm(N)
    t, v = int(0.7 * N), int(0.85 * N)
    return data[perm[:t]], data[perm[t:v]], data[perm[v:]], y[idx[perm.numpy()]]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _nmi_on_batch(model, loader, device):
    model.eval()
    with torch.no_grad():
        for batch, in loader:
            batch = batch.to(device)
            _, _, z = model(batch)
            if model.diffusion_extrema is not None:
                z = z[len(model.diffusion_extrema):]
            labels = torch.argmax(torch.softmax(z, dim=1), dim=1).cpu().numpy()
            half = len(labels) // 2
            return normalized_mutual_info_score(labels[:half], labels[half:half * 2])


def evaluate_recon(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch, in loader:
            batch = batch.to(device)
            recon, *_ = model(batch)
            total += ((recon - batch) ** 2).mean().item() * len(batch)
    return total / len(loader.dataset)


def sweep_archetypes(X_train, X_val, input_shape, device, n_arc_range, epochs):
    """Train AAnet for each k, return val losses, NMI, and best k."""
    train_loader = DataLoader(TensorDataset(X_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val),   batch_size=256, shuffle=False)

    reconstruction_losses, NMI_results = {}, {}
    best_k, best_val = None, float('inf')

    for k in tqdm(range(*n_arc_range), desc='Sweeping k'):
        lap_ext = get_laplacian_extrema(X_train.numpy(), n_extrema=k, knn=10,
                                        subsample=True)
        model = AAnet_vanilla(
            noise=0.05, layer_widths=[256, 128], n_archetypes=k,
            input_shape=input_shape, device=device,
            diffusion_extrema=X_train[lap_ext].float()
        ).float()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        nmi_list = []
        for epoch in range(1, epochs + 1):
            train_epoch(model, train_loader, opt, epoch=epoch,
                        gamma_reconstruction=1.0, gamma_archetypal=1.0,
                        gamma_extrema=1e-4, gamma_mi=0.1)
            nmi_list.append(_nmi_on_batch(model, train_loader, device))

        val_loss = evaluate_recon(model, val_loader, device)
        reconstruction_losses[k] = val_loss
        NMI_results[k] = nmi_list
        if val_loss < best_val:
            best_val, best_k = val_loss, k

    return reconstruction_losses, NMI_results, best_k


def repeated_runs(X_all, input_shape, device, best_k, R, epochs):
    """Train R times with best_k, compute consistency/ISI/NMI, save state dicts."""
    final_loader = DataLoader(TensorDataset(X_all), batch_size=256, shuffle=True)
    _, mSST = preprocess(X_all.numpy())

    archetype_list, NMI_list, state_dicts = [], [], []

    for r in tqdm(range(R), desc=f'Consistency runs (k={best_k})'):
        lap_ext = get_laplacian_extrema(X_all.numpy(), n_extrema=best_k)
        model = AAnet_vanilla(
            noise=0.05, layer_widths=[256, 128], n_archetypes=best_k,
            input_shape=input_shape, device=device,
            diffusion_extrema=X_all[lap_ext].float()
        ).float()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, epochs + 1):
            train_epoch(model, final_loader, opt, epoch=epoch,
                        gamma_reconstruction=1.0, gamma_archetypal=1.0,
                        gamma_extrema=1e-4, gamma_mi=0.1)

        NMI_list.append(_nmi_on_batch(model, final_loader, device))

        with torch.no_grad():
            XC = model.decode(model.archetypal_simplex).cpu().numpy().T  # (features, k)
        archetype_list.append(XC)
        state_dicts.append({k: v.cpu() for k, v in model.state_dict().items()})

    consistency_matrix = np.eye(R)
    ISI_matrix = np.eye(R)
    for i in range(R):
        for j in range(i + 1, R):
            c, isi = ArchetypeConsistency(archetype_list[i], archetype_list[j], mSST)
            consistency_matrix[i, j] = consistency_matrix[j, i] = c
            ISI_matrix[i, j] = ISI_matrix[j, i] = isi

    print(f"  Mean consistency: {consistency_matrix[np.triu_indices(R, k=1)].mean():.4f}")
    print(f"  Mean ISI:         {ISI_matrix[np.triu_indices(R, k=1)].mean():.4f}")
    print(f"  Mean NMI:         {np.mean(NMI_list):.4f}")
    return consistency_matrix, ISI_matrix, NMI_list, archetype_list, state_dicts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='AAnet training script')
    parser.add_argument('--dataset', required=True, choices=['mnist', 'blood'],
                        help='Dataset to train on')
    parser.add_argument('--subset', type=float, default=None,
                        help='Fraction of per-class data to use. Default: 1.0')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_arc_min', type=int, default=2)
    parser.add_argument('--n_arc_max', type=int, default=20)
    parser.add_argument('--R', type=int, default=5,
                        help='Repeated runs for consistency evaluation')
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()

    if args.subset is None:
        args.subset = 0.1

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    n_arc_range = (args.n_arc_min, args.n_arc_max)

    print(f'\n====  {args.dataset.upper()}  (subset={args.subset}) ====')
    loader_fn = load_mnist if args.dataset == 'mnist' else load_blood
    X_train, X_val, X_test, y = loader_fn(args.subset)
    input_shape = X_train.shape[1]

    print('Sweeping archetype counts...')
    recon_losses, NMI_results, best_k = sweep_archetypes(
        X_train, X_val, input_shape, device, n_arc_range, args.epochs)
    print(f'Best k = {best_k}  (val loss = {recon_losses[best_k]:.5f})')

    print('Repeated final runs...')
    X_all = torch.cat([X_train, X_val], dim=0)
    consistency_matrix, ISI_matrix, NMI_list, archetype_list, state_dicts = \
        repeated_runs(X_all, input_shape, device, best_k, args.R, args.epochs)

    result = {
        'reconstruction_losses': recon_losses,
        'NMI_results':           NMI_results,
        'best_k':                best_k,
        'consistency_matrix':    consistency_matrix,
        'ISI_matrix':            ISI_matrix,
        'NMI_list_final':        NMI_list,
        'archetype_list':        archetype_list,
        'model_state_dicts':     state_dicts,
        'input_shape':           input_shape,
        'layer_widths':          [256, 128],
        'labels':                y,
        'subset':                args.subset,
        'dataset':               args.dataset,
    }

    out_path = os.path.join(args.outdir, f'aanet_{args.dataset}_results.pt')
    torch.save(result, out_path)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
