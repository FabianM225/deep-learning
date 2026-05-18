"""
scAAnet (Single-cell Archetypal Analysis Network) training script.

scAAnet uses a count-distribution autoencoder (ZINB / NB / Poisson) to
discover archetypes in high-dimensional count data.

  paul15      → ae_type='zinb'  (zero-inflated NB; raw scRNA-seq counts)
  neurips2021 → ae_type='zinb'  (zero-inflated NB; raw scRNA-seq counts for 2000 HVGs)
  mnist       → ae_type='nb'    (NB; raw pixel values [0-255] treated as counts)
  blood       → ae_type='nb'    (same)

Usage:
    python scAAnet.py --dataset paul15
    python scAAnet.py --dataset neurips2021
    python scAAnet.py --dataset mnist  --K_consistency 10
    python scAAnet.py --dataset blood  --K_consistency 8

Saves: results/scaanet_{dataset}_results.pt

Dependencies:
    pip install scAAnet scanpy anndata
    (see requirements_scaanet.txt for the full list)
    For neurips2021: place GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad
    in ./data/ or install kagglehub for automatic download.

Note: scAAnet requires TensorFlow 2, which is incompatible with Python 3.12+.
Use a dedicated Python 3.10 / 3.11 environment for this script.
"""

import argparse
import itertools
import os

import numpy as np
import torch
from tqdm import tqdm

from utils import ArchetypeConsistency, preprocess, calcNMI


# ---------------------------------------------------------------------------
# Data loading  — scAAnet expects raw (non-normalised) count-like input
# ---------------------------------------------------------------------------

def load_mnist(subset):
    from torchvision import datasets
    mnist = datasets.MNIST(root='./data', train=True, download=True)
    X = mnist.data.numpy().reshape(60000, -1)   # (60000, 784), uint8
    y = mnist.targets.numpy()
    rng = np.random.default_rng(42)
    idx = np.hstack([rng.choice(np.where(y == d)[0],
                                max(1, int(subset * np.sum(y == d))), replace=False)
                     for d in np.unique(y)])
    return X[idx].astype(np.float32), y[idx], y[idx].astype(str), None


def load_blood(subset):
    from medmnist import BloodMNIST
    blood = BloodMNIST(split='train', download=True, size=28)
    X = blood.imgs.reshape(len(blood.imgs), -1)
    y = blood.labels.squeeze().astype(int)
    rng = np.random.default_rng(42)
    idx = np.hstack([rng.choice(np.where(y == d)[0],
                                max(1, int(subset * np.sum(y == d))), replace=False)
                     for d in np.unique(y)])
    return X[idx].astype(np.float32), y[idx], y[idx].astype(str), None


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

    # Select HVGs on normalized copy, extract raw counts for those genes (same as paul15)
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    sc.pp.highly_variable_genes(adata_norm, n_top_genes=2000)
    hvg_mask = adata_norm.var['highly_variable'].values
    gene_names = adata.var_names[hvg_mask].tolist()

    X_raw = adata.X
    if sp.issparse(X_raw):
        X_raw = X_raw.toarray()
    X = np.array(X_raw, dtype=np.float32)[:, hvg_mask]
    return X, y, labels_raw, gene_names


def load_paul15(subset):
    import scanpy as sc
    import scipy.sparse as sp

    adata = sc.datasets.paul15()

    # Select HVGs on normalised data, then extract RAW counts for those genes.
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    sc.pp.highly_variable_genes(adata_norm, n_top_genes=1000)
    hvg_mask   = adata_norm.var['highly_variable'].values
    gene_names = adata.var_names[hvg_mask].tolist()

    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X[:, hvg_mask].astype(np.float32)   # raw counts for HVGs

    labels_raw    = adata.obs['paul15_clusters'].astype(str).values
    unique_labels = sorted(set(labels_raw))
    label_map     = {l: i for i, l in enumerate(unique_labels)}
    y             = np.array([label_map[l] for l in labels_raw], dtype=int)

    rng = np.random.default_rng(42)
    idx = (np.hstack([rng.choice(np.where(y == d)[0],
                                  max(1, int(subset * np.sum(y == d))), replace=False)
                      for d in np.unique(y)])
           if subset < 1.0 else np.arange(len(y)))

    return X[idx], y[idx], labels_raw[idx], gene_names


# ---------------------------------------------------------------------------
# scAAnet fit wrapper
# ---------------------------------------------------------------------------

def fit_scaanet(X, K, ae_type, epochs, batch_size, lr):
    from scAAnet.api import scAAnet as run_scaanet
    re = run_scaanet(X, hidden_size=(128, K, 128), ae_type=ae_type,
                     epochs=epochs, batch_size=batch_size,
                     early_stop=50, reduce_lr=10, learning_rate=lr)
    usage   = np.array(re['usage'])    # (N, K) — per-cell mixing weights
    spectra = np.array(re['spectra'])  # (K, G) — archetype expression profiles
    recon   = np.array(re['recon'])    # (N, G) — reconstructed counts
    return usage, spectra, recon


# ---------------------------------------------------------------------------
# Sweep + consistency evaluation
# ---------------------------------------------------------------------------

def run_sweep(X, ae_type, K_list, n_runs, K_consistency, R,
              epochs, batch_size, lr, name):
    print(f"\n--- Loss / NMI sweep ({name}) ---")
    n_configs   = len(K_list)
    ReconLosses = np.zeros((n_configs, n_runs))
    Usages      = np.zeros((n_configs, n_runs), dtype=object)

    for i, K in enumerate(tqdm(K_list, desc=f'{name} sweep')):
        for r in range(n_runs):
            usage, _, recon = fit_scaanet(X, K, ae_type, epochs, batch_size, lr)
            ReconLosses[i, r] = float(np.mean((X - recon) ** 2))
            Usages[i, r]      = usage.T   # (K, N) — NMI convention

    pairs = np.array(list(itertools.combinations(range(n_runs), 2)))
    NMI   = np.zeros((n_configs, len(pairs)))
    for i in tqdm(range(n_configs), desc=f'{name} NMI'):
        for j, (a, b) in enumerate(pairs):
            NMI[i, j] = calcNMI(np.asarray(Usages[i, a]), np.asarray(Usages[i, b]))
    print(f"  Mean NMI: {NMI.mean():.4f}")

    print(f"--- Consistency runs K={K_consistency} ({name}) ---")
    _, mSST = preprocess(X)   # X is (N, features)
    archetype_list, usage_list = [], []
    final_recon = None

    for run_idx in tqdm(range(R), desc=f'{name} consistency'):
        usage_r, spectra_r, recon_r = fit_scaanet(X, K_consistency, ae_type,
                                                    epochs, batch_size, lr)
        archetype_list.append(spectra_r.T)  # (G, K) = (features, K)
        usage_list.append(usage_r)           # (N, K)
        if run_idx == 0:
            final_recon = recon_r

    consistency_matrix = np.eye(R)
    ISI_matrix         = np.eye(R)
    for i in range(R):
        for j in range(i + 1, R):
            c, isi = ArchetypeConsistency(archetype_list[i], archetype_list[j], mSST)
            consistency_matrix[i, j] = consistency_matrix[j, i] = c
            ISI_matrix[i, j]         = ISI_matrix[j, i]         = isi

    print(f"  Mean consistency: {consistency_matrix[np.triu_indices(R, k=1)].mean():.4f}")
    print(f"  Mean ISI:         {ISI_matrix[np.triu_indices(R, k=1)].mean():.4f}")

    return {
        'ReconLosses':        ReconLosses,
        'Usages':             Usages,          # (n_configs, n_runs) of (K, N)
        'NMI':                NMI,
        'K_list':             K_list,
        'pairs':              pairs,
        'consistency_matrix': consistency_matrix,
        'ISI_matrix':         ISI_matrix,
        'archetype_list':     archetype_list,  # R × (G, K)
        'usage_list':         usage_list,      # R × (N, K)
        'final_recon':        final_recon,     # (N, G) from first consistency run
        'mSST':               float(mSST),
        'K_consistency':      K_consistency,
        'R':                  R,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='scAAnet training script')
    parser.add_argument('--dataset', required=True,
                        choices=['mnist', 'blood', 'paul15', 'neurips2021'])
    parser.add_argument('--subset', type=float, default=None,
                        help='Fraction of per-class data. Default: 0.1 (mnist/blood), 1.0 (paul15)')
    parser.add_argument('--K_min', type=int, default=4)
    parser.add_argument('--K_max', type=int, default=16)
    parser.add_argument('--n_runs', type=int, default=4,
                        help='Repeated runs per K in sweep')
    parser.add_argument('--R', type=int, default=5,
                        help='Repeated runs for consistency evaluation')
    parser.add_argument('--K_consistency', type=int, default=None,
                        help='K for consistency runs. Default: 10 (mnist/paul15), 8 (blood)')
    parser.add_argument('--ae_type', type=str, default=None,
                        help='Loss distribution: zinb, nb, poisson, zipoisson. '
                             'Default: zinb (paul15), nb (mnist/blood)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()

    defaults = {
        'mnist':       {'subset': 0.1, 'K_consistency': 10, 'ae_type': 'nb'},
        'blood':       {'subset': 0.4, 'K_consistency': 8,  'ae_type': 'nb'},
        'paul15':      {'subset': 1.0, 'K_consistency': 10, 'ae_type': 'zinb'},
        'neurips2021': {'subset': 1.0, 'K_consistency': 10, 'ae_type': 'zinb'},
    }
    if args.subset is None:
        args.subset = defaults[args.dataset]['subset']
    if args.K_consistency is None:
        args.K_consistency = defaults[args.dataset]['K_consistency']
    if args.ae_type is None:
        args.ae_type = defaults[args.dataset]['ae_type']

    os.makedirs(args.outdir, exist_ok=True)
    K_list = list(range(args.K_min, args.K_max))

    print(f'\n====  {args.dataset.upper()}  '
          f'(subset={args.subset}, ae_type={args.ae_type}) ====')

    loaders = {
        'mnist':       load_mnist,
        'blood':       load_blood,
        'paul15':      load_paul15,
        'neurips2021': load_neurips2021,
    }
    X, y, label_names, gene_names = loaders[args.dataset](args.subset)

    names = {
        'mnist':       'MNIST',
        'blood':       'Blood',
        'paul15':      'Paul15',
        'neurips2021': 'NeurIPS 2021',
    }
    result = run_sweep(X, args.ae_type, K_list, args.n_runs, args.K_consistency,
                       args.R, args.epochs, args.batch_size, args.lr,
                       names[args.dataset])

    result['labels']      = y
    result['label_names'] = label_names
    result['subset']      = args.subset
    result['dataset']     = args.dataset
    result['ae_type']     = args.ae_type
    if gene_names is not None:
        result['gene_names'] = gene_names

    out_path = os.path.join(args.outdir, f'scaanet_{args.dataset}_results.pt')
    torch.save(result, out_path)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
