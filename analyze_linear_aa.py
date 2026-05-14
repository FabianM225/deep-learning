"""
Analysis script for Linear AA results.

Usage:
    python analyze_linear_aa.py --dataset mnist
    python analyze_linear_aa.py --dataset blood --k_star 8

Generates and saves:
  1. Loss & NMI stability curves vs number of archetypes
  2. Archetype image grids
  3. Latent-space UMAP with archetype positions marked
  4. Consistency & ISI heatmaps
  5. Original vs reconstruction image grid
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    import umap
    USE_UMAP = True
except ImportError:
    USE_UMAP = False
    print('umap-learn not found — falling back to PCA for latent space.')

CMAP  = {'mnist': 'gray_r', 'blood': None}
SHAPE = {'mnist': (28, 28),  'blood': (28, 28, 3)}
LABEL = {'mnist': 'MNIST',   'blood': 'BloodMNIST'}


def reload_X(ds, subset):
    """Return (features × samples) float64 in [0,1] matching training indices."""
    rng = np.random.default_rng(42)
    if ds == 'mnist':
        from torchvision import datasets, transforms
        mnist = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transforms.ToTensor())
        X = mnist.data.numpy().reshape(60000, -1)
        y = mnist.targets.numpy()
        idx = np.hstack([rng.choice(np.where(y == d)[0],
                                    max(1, int(subset * np.sum(y == d))),
                                    replace=False)
                         for d in np.unique(y)])
        return (X[idx] / 255.0).T.astype(np.float64), y[idx]   # (784, N)
    else:
        from medmnist import BloodMNIST
        blood = BloodMNIST(split='train', download=True, size=28)
        X = blood.imgs.reshape(len(blood.imgs), -1)
        y = blood.labels.squeeze().astype(int)
        idx = np.hstack([rng.choice(np.where(y == d)[0],
                                    max(1, int(subset * np.sum(y == d))),
                                    replace=False)
                         for d in np.unique(y)])
        return (X[idx].astype(np.float64) / 255.0).T, y[idx]   # (2352, N)


def main():
    parser = argparse.ArgumentParser(description='Analyze Linear AA results')
    parser.add_argument('--dataset', required=True, choices=['mnist', 'blood'])
    parser.add_argument('--k_star', type=int, default=None,
                        help='Override k* for plots. Default: use n_arc_consistency from results.')
    parser.add_argument('--results_path', default=None,
                        help='Path to results .pt file. Default: results/linear_aa_{dataset}_results.pt')
    parser.add_argument('--outdir', default='results/figures')
    args = parser.parse_args()

    if args.results_path is None:
        args.results_path = f'results/linear_aa_{args.dataset}_results.pt'

    os.makedirs(args.outdir, exist_ok=True)
    result = torch.load(args.results_path, weights_only=False)
    ds     = args.dataset
    subset = result['subset']
    prefix = f'linear_aa_{ds}'

    k_star = args.k_star if args.k_star is not None else result['n_arc_consistency']
    assert k_star in result['n_arc_list'], \
        f'--k_star={k_star} not in sweep range {result["n_arc_list"]}'

    # -------------------------------------------------------------------------
    # 1. Loss & NMI sweep curves
    # -------------------------------------------------------------------------

    ks       = result['n_arc_list']
    mean_nmi = result['NMI'].mean(axis=1)
    std_nmi  = result['NMI'].std(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle(f'Linear AA — {LABEL[ds]} — Sweep Analysis', fontsize=13, fontweight='bold')

    mean_val = result['ValLosses'].mean(axis=1)
    std_val  = result['ValLosses'].std(axis=1)
    axes[0].plot(ks, mean_val, 'b-o', ms=4, lw=1.5)
    axes[0].fill_between(ks, mean_val - std_val, mean_val + std_val, alpha=0.25, color='b')
    axes[0].axvline(k_star, ls='--', color='gray', lw=1.2, label=f'k*={k_star}')
    axes[0].set_title('Reconstruction Loss vs k')
    axes[0].set_xlabel('Number of archetypes k')
    axes[0].set_ylabel('Val SSE')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, mean_nmi, 'r-o', ms=4, lw=1.5)
    axes[1].fill_between(ks, mean_nmi - std_nmi, mean_nmi + std_nmi, alpha=0.25, color='r')
    axes[1].axvline(k_star, ls='--', color='gray', lw=1.2, label=f'k*={k_star}')
    axes[1].set_title('NMI Stability vs k')
    axes[1].set_xlabel('Number of archetypes k')
    axes[1].set_ylabel('Mean NMI across run pairs')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.outdir, f'{prefix}_loss_nmi_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.show()

    # -------------------------------------------------------------------------
    # 2. Archetype image grids
    # -------------------------------------------------------------------------

    XC    = np.array(result['archetype_list'][0])  # (features, k)
    k     = XC.shape[1]
    shape = SHAPE[ds]

    fig, axes = plt.subplots(1, k, figsize=(2 * k, 2.5))
    fig.suptitle(f'{LABEL[ds]} Archetypes (k={k})', fontsize=12, fontweight='bold')

    for i, ax in enumerate(axes):
        img = np.clip(XC[:, i].reshape(shape), 0, 1)
        ax.imshow(img, cmap=CMAP[ds])
        ax.axis('off')
        ax.set_title(f'A{i + 1}', fontsize=9)

    plt.tight_layout()
    path = os.path.join(args.outdir, f'{prefix}_archetypes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.show()

    # -------------------------------------------------------------------------
    # 3. Latent-space UMAP of S with archetype corners marked
    # -------------------------------------------------------------------------

    k_idx    = result['n_arc_list'].index(k_star)
    S_T      = np.asarray(result['Ss'][k_idx, 0]).T  # (n_samples, k)
    labels   = result['labels']
    if 'train_idx' in result:
        labels = labels[result['train_idx']]
    corners  = np.eye(k_star)
    combined = np.vstack([S_T, corners])

    method = 'UMAP' if USE_UMAP else 'PCA'
    if USE_UMAP:
        reducer  = umap.UMAP(n_components=2, random_state=42, min_dist=0.1)
        emb_all  = reducer.fit_transform(combined)
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'
    else:
        reducer  = PCA(n_components=2)
        emb_all  = reducer.fit_transform(combined)
        ev       = reducer.explained_variance_ratio_
        xlabel   = f'PC1 ({ev[0]*100:.1f} %)'
        ylabel   = f'PC2 ({ev[1]*100:.1f} %)'

    emb     = emb_all[:len(S_T)]
    arc_pos = emb_all[len(S_T):]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Linear AA — {LABEL[ds]} — Latent Space ({method})',
                 fontsize=13, fontweight='bold')
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10',
                         alpha=0.55, s=8, linewidths=0)
    plt.colorbar(scatter, ax=ax, label='class')
    ax.scatter(arc_pos[:, 0], arc_pos[:, 1],
               marker='X', c='red', s=120, zorder=5, label='Archetypes')
    for i, (x, y_) in enumerate(arc_pos):
        ax.annotate(f'A{i+1}', (x, y_), fontsize=7, color='red',
                    xytext=(3, 3), textcoords='offset points')
    ax.set_title(f'{LABEL[ds]}  k={k_star}', fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, markerscale=1.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.outdir, f'{prefix}_latent_space.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.show()

    # -------------------------------------------------------------------------
    # 4. Consistency & ISI heatmaps (side by side)
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Linear AA — {LABEL[ds]} — Consistency & ISI', fontsize=13, fontweight='bold')

    for col, (mat_key, title) in enumerate([('consistency_matrix', 'Consistency'),
                                             ('ISI_matrix',          'ISI')]):
        mat = result[mat_key]
        ax  = axes[col]
        im  = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
        plt.colorbar(im, ax=ax)
        R = mat.shape[0]
        for i in range(R):
            for j in range(R):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if mat[i, j] < 0.5 else 'black')
        ax.set_title(f'{LABEL[ds]} — {title}')
        ax.set_xlabel('Run')
        ax.set_ylabel('Run')

    plt.tight_layout()
    path = os.path.join(args.outdir, f'{prefix}_consistency_isi.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.show()

    # -------------------------------------------------------------------------
    # 5. Original vs reconstruction grid (one column per class)
    # -------------------------------------------------------------------------

    k_recon = result['n_arc_consistency']
    k_idx   = result['n_arc_list'].index(k_recon)
    XC      = np.array(result['archetype_list'][0])   # (features, k)
    S       = np.asarray(result['Ss'][k_idx, 0])      # (k_recon, n_samples)

    X_orig, y_orig = reload_X(ds, subset)
    if 'train_idx' in result:
        X_orig = X_orig[:, result['train_idx']]
        y_orig = y_orig[result['train_idx']]
    X_hat = XC @ S  # (features, n_train)

    classes   = np.unique(y_orig)
    n_classes = len(classes)
    shape     = SHAPE[ds]

    fig, axes = plt.subplots(2, n_classes, figsize=(1.8 * n_classes, 4))
    fig.suptitle(f'{LABEL[ds]} — Original (top) vs Reconstruction (bottom), k={k_recon}',
                 fontsize=11, fontweight='bold')

    for col_i, cls in enumerate(classes):
        sample_idx = np.where(y_orig == cls)[0][0]
        orig  = np.clip(X_orig[:, sample_idx].reshape(shape), 0, 1)
        recon = np.clip(X_hat[:,  sample_idx].reshape(shape), 0, 1)
        axes[0, col_i].imshow(orig,  cmap=CMAP[ds])
        axes[1, col_i].imshow(recon, cmap=CMAP[ds])
        axes[0, col_i].set_title(f'cls {cls}', fontsize=8)
        for row_i in range(2):
            axes[row_i, col_i].axis('off')

    axes[0, 0].set_ylabel('Original',       fontsize=9)
    axes[1, 0].set_ylabel('Reconstruction', fontsize=9)
    for row_i in range(2):
        axes[row_i, 0].axis('on')
        axes[row_i, 0].set_xticks([])
        axes[row_i, 0].set_yticks([])
        for spine in axes[row_i, 0].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(args.outdir, f'{prefix}_reconstructions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.show()

    print(f'\nAll figures saved to {args.outdir}/')


if __name__ == '__main__':
    main()
