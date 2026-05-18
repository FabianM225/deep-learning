import argparse
import itertools
import os
import numpy as np
import torch
from tqdm import tqdm
from LinearAAOrig.AALS import AALS, S_updateTorch
from utils import ArchetypeConsistency, preprocess, calcNMI
from torchvision import datasets, transforms
from medmnist import BloodMNIST

VAL_FRACTION = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paul15(subset):
    import scanpy as sc
    import scipy.sparse as sp
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var['highly_variable']].copy()
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float64)
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X = (X - X_min) / np.where(X_max - X_min > 0, X_max - X_min, 1.0)
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
    X_v1 = torch.from_numpy(X[idx].T)   # (features, N), double
    return X_v1, y[idx], labels_raw[idx], gene_names


def load_mnist(subset):
    mnist = datasets.MNIST(root='./data', train=True, download=True,
                           transform=transforms.ToTensor())
    X = mnist.data.numpy().reshape(60000, -1)
    y = mnist.targets.numpy()
    rng = np.random.default_rng(42)
    idx = np.hstack([rng.choice(np.where(y == d)[0],
                                max(1, int(subset * np.sum(y == d))), replace=False)
                     for d in np.unique(y)])
    X_sub = X[idx]
    X_v1 = torch.from_numpy((X_sub / 255.0).T).double()
    return X_v1, y[idx]


def load_blood(subset):
    blood = BloodMNIST(split='train', download=True, size=28)
    X = blood.imgs.reshape(len(blood.imgs), -1)
    y = blood.labels.squeeze()
    rng = np.random.default_rng(42)
    idx = np.hstack([rng.choice(np.where(y == d)[0],
                                max(1, int(subset * np.sum(y == d))), replace=False)
                     for d in np.unique(y)])
    X_sub = X[idx]
    X_v1 = torch.from_numpy((X_sub / 255.0).T).double()
    return X_v1, y[idx]


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

def compute_val_loss(A, X_val, n_iter=50, device='cpu'):
    k = A.shape[1]
    n_val = X_val.shape[1]
    S_val = torch.ones(k, n_val, dtype=torch.double, device=device) / k
    SSt = S_val @ S_val.T
    AtA = A.T @ A
    AtX = A.T @ X_val
    for _ in range(n_iter):
        S_val, SSt = S_updateTorch(S_val, AtA, AtX, SSt, k, n_val, device=device)
    X_recon = A @ S_val
    return torch.sum((X_val - X_recon) ** 2).item()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def run_dataset(X, n_arc_list, n_runs, n_arc_consistency, R, name,
                device=DEVICE, val_fraction=VAL_FRACTION):
    X = X.to(device)

    n_total = X.shape[1]
    n_val = max(1, int(val_fraction * n_total))
    perm = torch.randperm(n_total)
    train_idx = perm[n_val:].numpy()
    val_idx   = perm[:n_val].numpy()
    X_train = X[:, perm[n_val:]]
    X_val   = X[:, perm[:n_val]]
    print(f"  {name}: {X_train.shape[1]} train / {X_val.shape[1]} val samples")

    n_configs = len(n_arc_list)
    TrainLosses = np.zeros((n_configs, n_runs))
    ValLosses   = np.zeros((n_configs, n_runs))
    Ss = np.zeros((n_configs, n_runs), dtype=object)

    for n in tqdm(n_arc_list, desc=f'{name} sweep'):
        idx = n_arc_list.index(n)
        for i in range(n_runs):
            C, S, L, _ = AALS(X_train, n, device=device)
            TrainLosses[idx, i] = L[-1]
            Ss[idx, i] = S.cpu()
            A = X_train @ C
            ValLosses[idx, i] = compute_val_loss(A, X_val, device=device)

    pairs = np.array(list(itertools.combinations(range(n_runs), 2)))
    NMI = np.zeros((n_configs, len(pairs)))
    for n in tqdm(n_arc_list, desc=f'{name} NMI'):
        for j, (a, b) in enumerate(pairs):
            S1 = np.asarray(Ss[n_arc_list.index(n), a])
            S2 = np.asarray(Ss[n_arc_list.index(n), b])
            NMI[n_arc_list.index(n), j] = calcNMI(S1, S2)

    print(f"--- Consistency runs k={n_arc_consistency} ({name}) ---")
    X_train_np = X_train.cpu().numpy()
    _, mSST = preprocess(X_train_np)
    archetype_list = []

    for _ in tqdm(range(R), desc=f'{name} consistency'):
        C_r, _, _, _ = AALS(X_train, n_arc_consistency, device=device)
        C_t = C_r.cpu().float()
        X_t = torch.tensor(X_train_np, dtype=torch.float32)
        XC  = torch.matmul(X_t, C_t).numpy()
        archetype_list.append(XC)

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
        'Losses':             TrainLosses,
        'ValLosses':          ValLosses,
        'Ss':                 Ss,
        'NMI':                NMI,
        'n_arc_list':         n_arc_list,
        'pairs':              pairs,
        'consistency_matrix': consistency_matrix,
        'ISI_matrix':         ISI_matrix,
        'archetype_list':     archetype_list,
        'mSST':               float(mSST),
        'n_arc_consistency':  n_arc_consistency,
        'R':                  R,
        'train_idx':          train_idx,
        'val_idx':            val_idx,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Linear AA training script')
    parser.add_argument('--dataset', required=True, choices=['mnist', 'blood', 'paul15'],
                        help='Dataset to train on')
    parser.add_argument('--subset', type=float, default=None,
                        help='Fraction of per-class data to use.')
    parser.add_argument('--n_arc_min', type=int, default=2)
    parser.add_argument('--n_arc_max', type=int, default=20)
    parser.add_argument('--n_runs', type=int, default=4,
                        help='Repeated runs per archetype count (loss/NMI sweep)')
    parser.add_argument('--R', type=int, default=5,
                        help='Repeated runs for consistency evaluation')
    parser.add_argument('--n_arc_consistency', type=int, default=None,
                        help='Archetypes for consistency runs. Default: 10 (mnist), 8 (blood)')
    parser.add_argument('--val_fraction', type=float, default=VAL_FRACTION,
                        help='Fraction of data held out for validation')
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()

    defaults = {
        'mnist':  {'subset': 0.1, 'n_arc_consistency': 10},
        'blood':  {'subset': 0.4, 'n_arc_consistency': 8},
        'paul15': {'subset': 1.0, 'n_arc_consistency': 10},
    }
    if args.subset is None:
        args.subset = defaults[args.dataset]['subset']
    if args.n_arc_consistency is None:
        args.n_arc_consistency = defaults[args.dataset]['n_arc_consistency']

    os.makedirs(args.outdir, exist_ok=True)
    n_arc_list = list(range(args.n_arc_min, args.n_arc_max))

    print(f'\n====  {args.dataset.upper()}  (subset={args.subset}) ====')
    if args.dataset == 'paul15':
        X, y, label_names, gene_names = load_paul15(args.subset)
        name = 'Paul15'
    else:
        loaders = {'mnist': load_mnist, 'blood': load_blood}
        X, y = loaders[args.dataset](args.subset)
        label_names = y.astype(str)
        gene_names  = None
        name = 'MNIST' if args.dataset == 'mnist' else 'Blood'

    result = run_dataset(X, n_arc_list, args.n_runs, args.n_arc_consistency,
                         args.R, name, val_fraction=args.val_fraction)
    result['labels']      = y
    result['label_names'] = label_names
    result['subset']      = args.subset
    result['dataset']     = args.dataset
    if gene_names is not None:
        result['gene_names'] = gene_names

    out_path = os.path.join(args.outdir, f'linear_aa_{args.dataset}_results.pt')
    torch.save(result, out_path)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
