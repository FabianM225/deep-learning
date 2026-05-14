"""
MIDAA (Multimodal Archetypal Analysis) training script.

Usage:
    python MIDAA.py --dataset mnist
    python MIDAA.py --dataset blood --subset 0.4 --steps 1500

Saves: results/midaa_{dataset}_results.pt

Dependencies:
    pip install midaa anndata scanpy umap-learn
"""

import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import anndata as ad
import midaa as maa

from utils import preprocess, ArchetypeConsistency, calcNMI

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# MLP Decoder
# ---------------------------------------------------------------------------

class MLPDecoder(nn.Module):
    """General image decoder — Sigmoid output, works for any flat image size."""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, output_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist(subset):
    import torchvision
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    X = mnist.data.numpy().astype(np.float32)
    y = mnist.targets.numpy().astype(int)
    rng = np.random.default_rng(42)
    idx = np.hstack([rng.choice(np.where(y == d)[0],
                                max(1, int(subset * np.sum(y == d))), replace=False)
                     for d in np.unique(y)])
    X_sub = (X[idx] / 255.0).reshape(len(idx), -1)
    y_sub = y[idx]
    adata = ad.AnnData(X_sub)
    adata.obs['label'] = y_sub.astype(str)
    return adata, y_sub


def load_blood(subset):
    from medmnist import BloodMNIST
    blood = BloodMNIST(split='train', download=True, size=28)
    X = blood.imgs.reshape(len(blood.imgs), -1)
    y = blood.labels.squeeze().astype(int)
    rng = np.random.default_rng(42)
    idx = np.hstack([rng.choice(np.where(y == d)[0],
                                max(1, int(subset * np.sum(y == d))), replace=False)
                     for d in np.unique(y)])
    X_sub = X[idx].astype(np.float32) / 255.0
    y_sub = y[idx]
    adata = ad.AnnData(X_sub)
    adata.obs['label'] = y_sub.astype(str)
    return adata, y_sub


# ---------------------------------------------------------------------------
# MIDAA fit + decoder training
# ---------------------------------------------------------------------------

def fit_midaa(X_dense, narchetypes, steps, lr, seed=3):
    N = X_dense.shape[0]
    res = maa.fit_MIDAA(
        input_matrix=[X_dense],
        normalization_factor=[np.ones(N)],
        input_types=['G'],
        loss_weights_reconstruction=[1.0],
        side_matrices=None,
        input_types_side=None,
        loss_weights_side=None,
        lr=lr,
        steps=steps,
        narchetypes=narchetypes,
        torch_seed=seed,
    )
    # fit_MIDAA sets torch.set_default_device globally — reset so DataLoader works
    torch.set_default_device('cpu')
    Z = res['inferred_quantities']['Z']
    A = res['inferred_quantities']['archetypes_inferred']
    S = res['inferred_quantities'].get('S_inferred', None)
    elbo = res.get('ELBO', None)
    return Z, A, S, elbo


def train_decoder(decoder, Z, X, epochs=40, batch_size=256, lr=1e-3):
    decoder = decoder.to(DEVICE)
    Z_t = torch.tensor(Z).float().to(DEVICE)
    X_t = torch.tensor(X).float().to(DEVICE)
    min_len = min(len(Z_t), len(X_t))
    Z_t, X_t = Z_t[:min_len], X_t[:min_len]

    opt = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Z_t, X_t), batch_size=batch_size, shuffle=True)
    history = []

    for ep in range(epochs):
        total = 0.0
        for z_b, x_b in loader:
            opt.zero_grad()
            loss = loss_fn(decoder(z_b), x_b)
            loss.backward()
            opt.step()
            total += loss.item()
        history.append(total)
        print(f'  Decoder epoch {ep + 1}/{epochs}  loss={total:.4f}')

    return history


# ---------------------------------------------------------------------------
# Archetype count sweep
# ---------------------------------------------------------------------------

def run_sweep(X_dense, name, output_dim, n_arc_list, n_runs, n_arc_consistency,
              R, steps, lr, decoder_epochs):
    print(f"\n--- ELBO/NMI sweep ({name}) ---")
    n_configs = len(n_arc_list)
    ELBOs = np.zeros((n_configs, n_runs))
    Zs = np.zeros((n_configs, n_runs), dtype=object)

    for i, n in enumerate(tqdm(n_arc_list, desc=f'{name} sweep')):
        for r in range(n_runs):
            Z, A_r, S, elbo = fit_midaa(X_dense, n, steps, lr, seed=i * n_runs + r)
            final_elbo = elbo[-1] if isinstance(elbo, (list, np.ndarray)) else elbo
            ELBOs[i, r] = final_elbo if final_elbo is not None else float('nan')
            # calcNMI requires non-negative inputs. Use S_inferred if available,
            # otherwise softmax-normalise Z as a proxy.
            if S is not None:
                mixing = S
            else:
                z_shift = Z - Z.max(axis=1, keepdims=True)
                exp_z = np.exp(z_shift)
                mixing = exp_z / exp_z.sum(axis=1, keepdims=True)
            Zs[i, r] = mixing.T  # (K, N) — matches LinearAA S convention

    pairs = np.array(list(itertools.combinations(range(n_runs), 2)))
    NMI = np.zeros((n_configs, len(pairs)))
    for i in tqdm(range(n_configs), desc=f'{name} NMI'):
        for j, (a, b) in enumerate(pairs):
            NMI[i, j] = calcNMI(np.asarray(Zs[i, a]), np.asarray(Zs[i, b]))
    print(f"  Mean NMI:         {NMI.mean():.4f}")

    print(f"--- Consistency runs k={n_arc_consistency} ({name}) ---")
    _, mSST = preprocess(X_dense)
    archetype_list = []
    final_decoder = None
    final_Z, final_A = None, None

    for run_idx in tqdm(range(R), desc=f'{name} consistency'):
        Z_r, A_r, _, _ = fit_midaa(X_dense, n_arc_consistency, steps, lr, seed=run_idx)
        dec = MLPDecoder(Z_r.shape[1], output_dim)
        train_decoder(dec, Z_r, X_dense, epochs=decoder_epochs)
        A_t = torch.tensor(A_r).float().to(DEVICE)
        with torch.no_grad():
            XC = dec(A_t).cpu().numpy().T
        archetype_list.append(XC)
        if run_idx == 0:
            final_decoder, final_Z, final_A = dec, Z_r, A_r

    consistency_matrix = np.eye(R)
    ISI_matrix = np.eye(R)
    for i in range(R):
        for j in range(i + 1, R):
            c, isi = ArchetypeConsistency(archetype_list[i], archetype_list[j], mSST)
            consistency_matrix[i, j] = consistency_matrix[j, i] = c
            ISI_matrix[i, j] = ISI_matrix[j, i] = isi

    print(f"  Mean consistency: {consistency_matrix[np.triu_indices(R, k=1)].mean():.4f}")
    print(f"  Mean ISI:         {ISI_matrix[np.triu_indices(R, k=1)].mean():.4f}")

    return {
        'ELBOs':               ELBOs,
        'Zs':                  Zs,
        'NMI':                 NMI,
        'n_arc_list':          n_arc_list,
        'pairs':               pairs,
        'consistency_matrix':  consistency_matrix,
        'ISI_matrix':          ISI_matrix,
        'archetype_list':      archetype_list,
        'mSST':                float(mSST),
        'n_arc_consistency':   n_arc_consistency,
        'R':                   R,
        'Z':                   final_Z,
        'A':                   final_A,
        'decoder_state_dict':  final_decoder.state_dict(),
        'latent_dim':          final_Z.shape[1],
        'input_dim':           X_dense.shape[1],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MIDAA training script')
    parser.add_argument('--dataset', required=True, choices=['mnist', 'blood'],
                        help='Dataset to train on')
    parser.add_argument('--subset', type=float, default=None,
                        help='Fraction of per-class data. Default: 0.2 (mnist), 0.4 (blood)')
    parser.add_argument('--n_arc_min', type=int, default=6)
    parser.add_argument('--n_arc_max', type=int, default=16)
    parser.add_argument('--n_runs', type=int, default=4,
                        help='Repeated runs per archetype count (ELBO/NMI sweep)')
    parser.add_argument('--R', type=int, default=5,
                        help='Repeated runs for consistency evaluation')
    parser.add_argument('--n_arc_consistency', type=int, default=None,
                        help='Archetypes for consistency runs. Default: 10 (mnist), 8 (blood)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='MIDAA optimisation steps')
    parser.add_argument('--lr_midaa', type=float, default=0.001)
    parser.add_argument('--decoder_epochs', type=int, default=40)
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()

    if args.subset is None:
        args.subset = 0.1
    if args.n_arc_consistency is None:
        args.n_arc_consistency = 8 if args.dataset == 'blood' else 10

    os.makedirs(args.outdir, exist_ok=True)
    print(f'Device: {DEVICE}' + (f'  ({torch.cuda.get_device_name(0)})' if DEVICE == 'cuda' else ''))
    n_arc_list = list(range(args.n_arc_min, args.n_arc_max))

    print(f'\n====  {args.dataset.upper()}  (subset={args.subset}) ====')
    loader_fn = load_mnist if args.dataset == 'mnist' else load_blood
    adata, y = loader_fn(args.subset)

    name = 'MNIST' if args.dataset == 'mnist' else 'Blood'
    output_dim = 784 if args.dataset == 'mnist' else 2352
    result = run_sweep(adata.X, name, output_dim, n_arc_list, args.n_runs,
                       args.n_arc_consistency, args.R,
                       args.steps, args.lr_midaa, args.decoder_epochs)
    result['labels']  = y
    result['subset']  = args.subset
    result['dataset'] = args.dataset

    out_path = os.path.join(args.outdir, f'midaa_{args.dataset}_results.pt')
    torch.save(result, out_path)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
