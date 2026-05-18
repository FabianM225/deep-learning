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

def load_neurips2021(subset):
    import scanpy as sc
    import scipy.sparse as sp
    fn = './data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad'
    if not os.path.exists(fn):
        try:
            import kagglehub
            import glob as _glob
            path = kagglehub.dataset_download(
                'alexandervc/scrnaseq-scatacseq-challenge-at-neurips-2021',
                force_download=False)
            candidates = _glob.glob(os.path.join(path, '**', '*.h5ad'), recursive=True)
            if candidates:
                fn = candidates[0]
        except Exception:
            raise FileNotFoundError(
                f'NeurIPS 2021 h5ad not found at {fn!r}. '
                'Place the file at ./data/ or install kagglehub for automatic download.')
    adata_full = sc.read(fn)
    n_obs = max(100, int(subset * 10000))
    adata = sc.pp.subsample(adata_full, n_obs=n_obs, copy=True, random_state=42)

    labels_raw = adata.obs['cell_type'].astype(str).values
    unique_labels = sorted(set(labels_raw))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels_raw], dtype=int)

    # Same HVG pipeline as paul15
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']].copy()

    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X = (X - X_min) / np.where(X_max - X_min > 0, X_max - X_min, 1.0)
    X = X * 2 - 1  # [0,1] → [-1,1]
    gene_names = adata.var_names.tolist()
    data = torch.from_numpy(X).float()
    N = len(data)
    perm = torch.randperm(N)
    perm_np = perm.numpy()
    t, v = int(0.7 * N), int(0.85 * N)
    return (data[perm[:t]], data[perm[t:v]], data[perm[v:]],
            y[perm_np], labels_raw[perm_np], gene_names)


def load_paul15(subset):
    import scanpy as sc
    import scipy.sparse as sp
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var['highly_variable']].copy()
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X = (X - X_min) / np.where(X_max - X_min > 0, X_max - X_min, 1.0)
    X = X * 2 - 1  # [0,1] → [-1,1]
    gene_names = adata.var_names.tolist()
    labels_raw = adata.obs['paul15_clusters'].astype(str).values
    unique_labels = sorted(set(labels_raw))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels_raw], dtype=int)
    rng = np.random.default_rng(42)
    idx = (np.hstack([rng.choice(np.where(y == d)[0],
                                  max(1, int(subset * np.sum(y == d))), replace=False)
                      for d in np.unique(y)])
           if subset < 1.0 else np.arange(len(y)))
    data = torch.from_numpy(X[idx]).float()
    N = len(data)
    perm = torch.randperm(N)
    t, v = int(0.7 * N), int(0.85 * N)
    perm_np = perm.numpy()
    return (data[perm[:t]], data[perm[t:v]], data[perm[v:]],
            y[idx][perm_np], labels_raw[idx][perm_np], gene_names)


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

    for _ in tqdm(range(R), desc=f'Consistency runs (k={best_k})'):
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
    parser.add_argument('--dataset', required=True,
                        choices=['mnist', 'blood', 'paul15', 'neurips2021'],
                        help='Dataset to train on')
    parser.add_argument('--subset', type=float, default=None,
                        help='Fraction of per-class data to use.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_arc_min', type=int, default=2)
    parser.add_argument('--n_arc_max', type=int, default=20)
    parser.add_argument('--R', type=int, default=5,
                        help='Repeated runs for consistency evaluation')
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()

    defaults = {
        'mnist':       {'subset': 0.1},
        'blood':       {'subset': 0.4},
        'paul15':      {'subset': 1.0},
        'neurips2021': {'subset': 1.0},
    }
    if args.subset is None:
        args.subset = defaults[args.dataset]['subset']

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    n_arc_range = (args.n_arc_min, args.n_arc_max)

    print(f'\n====  {args.dataset.upper()}  (subset={args.subset}) ====')
    if args.dataset == 'paul15':
        X_train, X_val, _, y, label_names, gene_names = load_paul15(args.subset)
    elif args.dataset == 'neurips2021':
        X_train, X_val, _, y, label_names, gene_names = load_neurips2021(args.subset)
    else:
        loaders = {'mnist': load_mnist, 'blood': load_blood}
        X_train, X_val, _, y = loaders[args.dataset](args.subset)
        label_names = y.astype(str)
        gene_names  = None
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
        'label_names':           label_names,
        'subset':                args.subset,
        'dataset':               args.dataset,
    }
    if gene_names is not None:
        result['gene_names'] = gene_names

    out_path = os.path.join(args.outdir, f'aanet_{args.dataset}_results.pt')
    torch.save(result, out_path)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
