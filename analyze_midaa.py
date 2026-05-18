"""
Analysis script for MIDAA results.

Usage:
    python analyze_midaa.py --dataset mnist
    python analyze_midaa.py --dataset blood --k_star 8
    python analyze_midaa.py --dataset paul15 --k_star 10
    python analyze_midaa.py --dataset neurips2021 --k_star 10

Generates and saves:
  1. ELBO & NMI stability curves vs number of archetypes
  2a. [mnist/blood]           Archetype image grids (decoded through MLP decoder)
  2b. [paul15/neurips2021]    Archetype feature expression heatmap
  3. Latent-space UMAP / PCA of Z (per-sample barycentric coordinates)
  4. Consistency & ISI heatmaps
  5. [mnist/blood] Original vs reconstruction image grid
  6. [paul15/neurips2021]     Archetype mixing weight distributions per cell type
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print('seaborn not found — mixing weight violin plots will be skipped.')

CMAP  = {'mnist': 'gray_r', 'blood': None, 'paul15': None, 'neurips2021': None}
SHAPE = {'mnist': (28, 28), 'blood': (28, 28, 3), 'paul15': None, 'neurips2021': None}
LABEL = {'mnist': 'MNIST',  'blood': 'BloodMNIST', 'paul15': 'Paul et al. 2015',
         'neurips2021': 'NeurIPS 2021 BMMC'}


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, output_activation=None):
        super().__init__()
        activation = output_activation if output_activation is not None else nn.Sigmoid()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, output_dim), activation,
        )

    def forward(self, z):
        return self.net(z)


def rebuild_decoder(result):
    decoder = MLPDecoder(result['latent_dim'], result['input_dim'])
    decoder.load_state_dict(result['decoder_state_dict'])
    decoder.eval()
    return decoder


def reload_X(ds, subset):
    rng = np.random.default_rng(42)
    if ds == 'mnist':
        from torchvision import datasets, transforms
        mnist = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transforms.ToTensor())
        X = mnist.data.numpy().reshape(60000, -1).astype(np.float32) / 255.0
        y = mnist.targets.numpy()
        idx = np.hstack([rng.choice(np.where(y == d)[0],
                                    max(1, int(subset * np.sum(y == d))),
                                    replace=False)
                         for d in np.unique(y)])
        return X[idx], y[idx]
    elif ds == 'blood':
        from medmnist import BloodMNIST
        blood = BloodMNIST(split='train', download=True, size=28)
        X = blood.imgs.reshape(len(blood.imgs), -1).astype(np.float32) / 255.0
        y = blood.labels.squeeze().astype(int)
        idx = np.hstack([rng.choice(np.where(y == d)[0],
                                    max(1, int(subset * np.sum(y == d))),
                                    replace=False)
                         for d in np.unique(y)])
        return X[idx], y[idx]


def _mixing_weights(Z):
    """Softmax rows of Z to get per-cell archetype weights summing to 1."""
    z_shift = Z - Z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description='Analyze MIDAA results')
    parser.add_argument('--dataset', required=True,
                        choices=['mnist', 'blood', 'paul15', 'neurips2021'])
    parser.add_argument('--k_star', type=int, default=None,
                        help='Override k* for plots. Default: use n_arc_consistency from results.')
    parser.add_argument('--results_path', default=None,
                        help='Path to results .pt file. Default: results/midaa_{dataset}_results.pt')
    parser.add_argument('--outdir', default='results/figures')
    args = parser.parse_args()

    if args.results_path is None:
        args.results_path = f'results/midaa_{args.dataset}_results.pt'

    os.makedirs(args.outdir, exist_ok=True)
    result = torch.load(args.results_path, weights_only=False)
    ds     = args.dataset
    subset = result['subset']
    prefix = f'midaa_{ds}'
    label  = LABEL[ds]

    k_star = args.k_star if args.k_star is not None else result['n_arc_consistency']
    assert k_star in result['n_arc_list'], \
        f'--k_star={k_star} not in sweep range {result["n_arc_list"]}'

    # -------------------------------------------------------------------------
    # 1. ELBO & NMI sweep curves
    # -------------------------------------------------------------------------

    ks        = result['n_arc_list']
    mean_elbo = result['ELBOs'].mean(axis=1)
    std_elbo  = result['ELBOs'].std(axis=1)
    mean_nmi  = result['NMI'].mean(axis=1)
    std_nmi   = result['NMI'].std(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle(f'MIDAA — {label} — Sweep Analysis', fontsize=13, fontweight='bold')

    axes[0].plot(ks, mean_elbo, 'b-o', ms=4, lw=1.5)
    axes[0].fill_between(ks, mean_elbo - std_elbo, mean_elbo + std_elbo, alpha=0.25, color='b')
    axes[0].axvline(k_star, ls='--', color='gray', lw=1.2, label=f'k*={k_star}')
    axes[0].set_title('ELBO vs k')
    axes[0].set_xlabel('Number of archetypes k')
    axes[0].set_ylabel('ELBO')
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
    path = os.path.join(args.outdir, f'{prefix}_elbo_nmi_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    # -------------------------------------------------------------------------
    # 2a. [mnist / blood] Archetype image grids
    # 2b. [paul15]        Archetype gene expression heatmap
    # -------------------------------------------------------------------------

    decoder = rebuild_decoder(result)
    A_t     = torch.tensor(np.array(result['A'])).float()

    with torch.no_grad():
        decoded = decoder(A_t).numpy()  # (k, features)

    k = decoded.shape[0]

    if ds in ('paul15', 'neurips2021'):
        top_n      = 30
        gene_var   = decoded.var(axis=0)
        top_idx    = np.argsort(gene_var)[-top_n:]
        heatmap    = decoded[:, top_idx]           # (k, top_n)
        gene_names = result.get('gene_names', [f'G{i}' for i in top_idx])
        col_labels = [gene_names[i] for i in top_idx]

        fig_h = max(4, 0.45 * k + 1.5)
        fig_w = max(12, top_n * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.suptitle(f'{label} — Archetype Gene Expression (top {top_n} discriminative genes)',
                     fontsize=12, fontweight='bold')

        im = ax.imshow(heatmap, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Decoded expression [0–1]', fraction=0.03)
        ax.set_yticks(range(k))
        ax.set_yticklabels([f'A{i + 1}' for i in range(k)], fontsize=9)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
        ax.set_ylabel('Archetype')
        ax.set_xlabel('Gene')

        plt.tight_layout()
        path = os.path.join(args.outdir, f'{prefix}_archetype_gene_heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
        plt.close()

    else:
        shape = SHAPE[ds]
        fig, axes = plt.subplots(1, k, figsize=(2 * k, 2.5))
        fig.suptitle(f'{label} Archetypes (k={k})', fontsize=12, fontweight='bold')

        for i, ax in enumerate(axes):
            img = np.clip(decoded[i].reshape(shape), 0, 1)
            ax.imshow(img, cmap=CMAP[ds])
            ax.axis('off')
            ax.set_title(f'A{i + 1}', fontsize=9)

        plt.tight_layout()
        path = os.path.join(args.outdir, f'{prefix}_archetypes.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
        plt.close()

    # -------------------------------------------------------------------------
    # 3. Latent-space UMAP / PCA of Z
    # -------------------------------------------------------------------------

    Z        = np.array(result['Z'])
    labels   = np.array(result['labels'])
    A_arr    = np.array(result['A'])
    combined = np.vstack([Z, A_arr])

    method = 'UMAP' if USE_UMAP else 'PCA'
    if USE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1)
        emb_all = reducer.fit_transform(combined)
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'
    else:
        reducer = PCA(n_components=2)
        emb_all = reducer.fit_transform(combined)
        ev      = reducer.explained_variance_ratio_
        xlabel  = f'PC1 ({ev[0]*100:.1f} %)'
        ylabel  = f'PC2 ({ev[1]*100:.1f} %)'

    emb     = emb_all[:len(Z)]
    arc_pos = emb_all[len(Z):]

    n_classes  = len(np.unique(labels))
    cmap_name  = 'tab20' if n_classes > 10 else 'tab10'

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'MIDAA — {label} — Latent Space ({method})', fontsize=13, fontweight='bold')

    if ds in ('paul15', 'neurips2021'):
        label_names = result.get('label_names', labels.astype(str))
        unique_ct   = sorted(set(label_names))
        colors      = plt.get_cmap(cmap_name)(np.linspace(0, 1, len(unique_ct)))
        ct_to_col   = {ct: colors[i] for i, ct in enumerate(unique_ct)}
        cell_colors = [ct_to_col[ln] for ln in label_names]
        ax.scatter(emb[:, 0], emb[:, 1], c=cell_colors, alpha=0.5, s=8, linewidths=0)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=ct_to_col[ct], markersize=6, label=ct)
                   for ct in unique_ct]
        ax.legend(handles=handles, fontsize=6, ncol=2, loc='best', title='Cell type')
    else:
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap_name,
                             alpha=0.55, s=8, linewidths=0)
        plt.colorbar(scatter, ax=ax, label='class')

    ax.scatter(arc_pos[:, 0], arc_pos[:, 1],
               marker='X', c='red', s=120, zorder=5, label='Archetypes')
    for i, (x_, y_) in enumerate(arc_pos):
        ax.annotate(f'A{i + 1}', (x_, y_), fontsize=7, color='red',
                    xytext=(3, 3), textcoords='offset points')
    ax.set_title(f'{label}  k={result["n_arc_consistency"]}', fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.outdir, f'{prefix}_latent_space.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    # -------------------------------------------------------------------------
    # 4. Consistency & ISI heatmaps (side by side)
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'MIDAA — {label} — Consistency & ISI', fontsize=13, fontweight='bold')

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
        ax.set_title(f'{label} — {title}')
        ax.set_xlabel('Run')
        ax.set_ylabel('Run')

    plt.tight_layout()
    path = os.path.join(args.outdir, f'{prefix}_consistency_isi.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    # -------------------------------------------------------------------------
    # 5. [mnist / blood] Original vs reconstruction image grid
    # -------------------------------------------------------------------------

    if ds not in ('paul15', 'neurips2021'):
        decoder   = rebuild_decoder(result)
        X, y_orig = reload_X(ds, subset)
        Z         = np.array(result['Z'])
        min_len   = min(len(Z), len(X))
        Z_t       = torch.tensor(Z[:min_len]).float()

        with torch.no_grad():
            recon = decoder(Z_t).numpy()

        classes   = np.unique(y_orig[:min_len])
        n_classes = len(classes)
        shape     = SHAPE[ds]
        k         = result['n_arc_consistency']

        fig, axes = plt.subplots(2, n_classes, figsize=(1.8 * n_classes, 4))
        fig.suptitle(f'{label} — Original (top) vs Reconstruction (bottom), k={k}',
                     fontsize=11, fontweight='bold')

        for col_i, cls in enumerate(classes):
            sample_idx = np.where(y_orig[:min_len] == cls)[0][0]
            orig = np.clip(X[sample_idx].reshape(shape), 0, 1)
            rec  = np.clip(recon[sample_idx].reshape(shape), 0, 1)
            axes[0, col_i].imshow(orig, cmap=CMAP[ds])
            axes[1, col_i].imshow(rec,  cmap=CMAP[ds])
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
        plt.close()

    # -------------------------------------------------------------------------
    # 6. [paul15] Archetype mixing weight distributions per cell type
    # -------------------------------------------------------------------------

    if ds in ('paul15', 'neurips2021'):
        if not HAS_SEABORN:
            print('Skipping mixing weight plot — seaborn not installed.')
        else:
            Z         = np.array(result['Z'])
            weights   = _mixing_weights(Z)          # (N, k)
            k         = weights.shape[1]
            label_names = result.get('label_names', result['labels'].astype(str))

            rows = [{'cell_type': label_names[n],
                     'archetype': f'A{arc + 1}',
                     'weight':    float(weights[n, arc])}
                    for n in range(len(weights))
                    for arc in range(k)]
            df = pd.DataFrame(rows)

            n_cols  = min(5, k)
            n_rows  = int(np.ceil(k / n_cols))
            cell_types = sorted(df['cell_type'].unique())

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(4.5 * n_cols, 3.5 * n_rows),
                                     sharey=True)
            fig.suptitle(f'{label} — Archetype Mixing Weights per Cell Type',
                         fontsize=13, fontweight='bold')
            axes_flat = np.array(axes).ravel()

            for arc_i in range(k):
                ax      = axes_flat[arc_i]
                arc_df  = df[df['archetype'] == f'A{arc_i + 1}']
                sns.violinplot(data=arc_df, x='cell_type', y='weight', ax=ax,
                               order=cell_types, inner='box', color='steelblue', cut=0)
                ax.set_title(f'A{arc_i + 1}', fontsize=10, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Weight' if arc_i % n_cols == 0 else '')
                ax.set_xticklabels(cell_types, rotation=45, ha='right', fontsize=7)
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, axis='y', alpha=0.3)

            for i in range(k, len(axes_flat)):
                axes_flat[i].set_visible(False)

            plt.tight_layout()
            path = os.path.join(args.outdir, f'{prefix}_mixing_weights.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f'Saved: {path}')
            plt.close()

    print(f'\nAll figures saved to {args.outdir}/')


if __name__ == '__main__':
    main()
