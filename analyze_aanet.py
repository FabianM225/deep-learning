"""
Analysis script for AAnet results.

Usage:
    python analyze_aanet.py --dataset mnist
    python analyze_aanet.py --dataset blood --k_star 8
    python analyze_aanet.py --dataset paul15

Generates and saves:
  1. Reconstruction loss vs k and final-epoch NMI vs k
  2a. [mnist/blood] Archetype image grids
  2b. [paul15]      Archetype gene expression heatmap
  3. Latent-space UMAP / PCA of encoder outputs with archetype corners
  4. Consistency & ISI heatmaps
  5. [mnist/blood] Original vs reconstruction image grid
  6. [paul15]      Archetype mixing weight distributions per cell type
"""

import argparse
import os
import torch
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

CMAP  = {'mnist': 'gray_r', 'blood': None,   'paul15': None}
SHAPE = {'mnist': (28, 28),  'blood': (28, 28, 3), 'paul15': None}
LABEL = {'mnist': 'MNIST',   'blood': 'BloodMNIST', 'paul15': 'Paul et al. 2015'}


def reload_data(ds, subset):
    """Return (N, features) float32 in [-1,1] and int labels."""
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
        return ((X[idx] / 255.0) * 2 - 1).astype(np.float32), y[idx]
    else:
        from medmnist import BloodMNIST
        blood = BloodMNIST(split='train', download=True, size=28)
        X = blood.imgs.reshape(len(blood.imgs), -1)
        y = blood.labels.squeeze().astype(int)
        idx = np.hstack([rng.choice(np.where(y == d)[0],
                                    max(1, int(subset * np.sum(y == d))),
                                    replace=False)
                         for d in np.unique(y)])
        return ((X[idx].astype(np.float32) / 255.0) * 2 - 1), y[idx]


def reload_paul15(subset):
    """Return (N, features) float32 in [-1,1], int labels, and string cell-type labels."""
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
    X = X * 2 - 1
    labels_raw = adata.obs['paul15_clusters'].astype(str).values
    unique_labels = sorted(set(labels_raw))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels_raw], dtype=int)
    rng = np.random.default_rng(42)
    idx = (np.hstack([rng.choice(np.where(y == d)[0],
                                  max(1, int(subset * np.sum(y == d))), replace=False)
                      for d in np.unique(y)])
           if subset < 1.0 else np.arange(len(y)))
    return X[idx], y[idx], labels_raw[idx]


def rebuild_model(result):
    from aanetOrig import AAnet_vanilla
    model = AAnet_vanilla(
        noise=0, layer_widths=result['layer_widths'], n_archetypes=result['best_k'],
        input_shape=result['input_shape']
    ).float()
    model.load_state_dict(result['model_state_dicts'][-1])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Analyze AAnet results')
    parser.add_argument('--dataset', required=True, choices=['mnist', 'blood', 'paul15'])
    parser.add_argument('--k_star', type=int, default=None,
                        help='Override k* for plots. Default: use best_k from results.')
    parser.add_argument('--results_path', default=None,
                        help='Path to results .pt file. Default: results/aanet_{dataset}_results.pt')
    parser.add_argument('--outdir', default='results/figures')
    args = parser.parse_args()

    if args.results_path is None:
        args.results_path = f'results/aanet_{args.dataset}_results.pt'

    os.makedirs(args.outdir, exist_ok=True)
    result = torch.load(args.results_path, weights_only=False)
    ds     = args.dataset
    subset = result['subset']
    k_star = args.k_star if args.k_star is not None else result['best_k']
    prefix = f'aanet_{ds}'
    label  = LABEL[ds]

    # -------------------------------------------------------------------------
    # 1. Reconstruction loss vs k  +  final-epoch NMI vs k
    # -------------------------------------------------------------------------

    ks     = sorted(result['reconstruction_losses'].keys())
    losses = [result['reconstruction_losses'][k] for k in ks]
    nmis   = [result['NMI_results'][k][-1] for k in ks]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle(f'AAnet — {label} — Sweep Analysis', fontsize=13, fontweight='bold')

    axes[0].plot(ks, losses, 'b-o', ms=4, lw=1.5)
    axes[0].axvline(k_star, ls='--', color='gray', lw=1.2, label=f'k*={k_star}')
    axes[0].set_title('Reconstruction Loss vs k')
    axes[0].set_xlabel('Number of archetypes k')
    axes[0].set_ylabel('Val MSE')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, nmis, 'r-o', ms=4, lw=1.5)
    axes[1].axvline(k_star, ls='--', color='gray', lw=1.2, label=f'k*={k_star}')
    axes[1].set_title('NMI (final epoch) vs k')
    axes[1].set_xlabel('Number of archetypes k')
    axes[1].set_ylabel('NMI')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.outdir, f'{prefix}_loss_nmi_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

    # -------------------------------------------------------------------------
    # 2a. [mnist / blood] Archetype image grids
    # 2b. [paul15]        Archetype gene expression heatmap
    # -------------------------------------------------------------------------

    XC = np.array(result['archetype_list'][0])   # (features, k) in [-1,1] space
    k  = XC.shape[1]

    if ds == 'paul15':
        decoded_01 = (XC.T + 1) / 2              # (k, n_genes), back to [0,1]
        top_n      = 30
        top_idx    = np.argsort(decoded_01.var(axis=0))[-top_n:]
        heatmap    = decoded_01[:, top_idx]
        gene_names  = result.get('gene_names', [f'G{i}' for i in top_idx])
        col_labels  = [gene_names[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(max(12, top_n * 0.35), max(4, 0.45 * k + 1.5)))
        fig.suptitle(f'{label} — Archetype Gene Expression (top {top_n} discriminative genes)',
                     fontsize=12, fontweight='bold')
        im = ax.imshow(heatmap, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Expression [0–1]', fraction=0.03)
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
            img = np.clip((XC[:, i].reshape(shape) + 1) / 2, 0, 1)
            ax.imshow(img, cmap=CMAP[ds])
            ax.axis('off')
            ax.set_title(f'A{i + 1}', fontsize=9)

        plt.tight_layout()
        path = os.path.join(args.outdir, f'{prefix}_archetypes.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
        plt.close()

    # -------------------------------------------------------------------------
    # 3. Latent-space UMAP / PCA of encoder outputs
    # -------------------------------------------------------------------------

    model = rebuild_model(result)

    if ds == 'paul15':
        X_data, labels, label_names_str = reload_paul15(subset)
    else:
        X_data, labels = reload_data(ds, subset)
        label_names_str = labels.astype(str)

    X_t = torch.tensor(X_data)
    with torch.no_grad():
        z = model.encode(X_t).numpy()

    arc_pts  = model.archetypal_simplex.cpu().numpy()
    combined = np.vstack([z, arc_pts])

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

    emb     = emb_all[:len(z)]
    arc_pos = emb_all[len(z):]

    n_classes = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'AAnet — {label} — Latent Space ({method})',
                 fontsize=13, fontweight='bold')

    if ds == 'paul15':
        unique_ct = sorted(set(label_names_str))
        colors    = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_ct)))
        ct_to_col = {ct: colors[i] for i, ct in enumerate(unique_ct)}
        cell_colors = [ct_to_col[ln] for ln in label_names_str]
        ax.scatter(emb[:, 0], emb[:, 1], c=cell_colors, alpha=0.5, s=8, linewidths=0)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=ct_to_col[ct], markersize=6, label=ct)
                   for ct in unique_ct]
        ax.legend(handles=handles, fontsize=6, ncol=2, loc='best', title='Cell type')
    else:
        cmap_name = 'tab20' if n_classes > 10 else 'tab10'
        scatter   = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap_name,
                               alpha=0.55, s=8, linewidths=0)
        plt.colorbar(scatter, ax=ax, label='class')

    ax.scatter(arc_pos[:, 0], arc_pos[:, 1],
               marker='X', c='red', s=120, zorder=5, label='Archetypes')
    for i, (x_, y_) in enumerate(arc_pos):
        ax.annotate(f'A{i + 1}', (x_, y_), fontsize=7, color='red',
                    xytext=(3, 3), textcoords='offset points')
    ax.set_title(f'{label}  k={result["best_k"]}', fontsize=11)
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
    fig.suptitle(f'AAnet — {label} — Consistency & ISI', fontsize=13, fontweight='bold')

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

    if ds != 'paul15':
        model     = rebuild_model(result)
        X_img, y_orig = reload_data(ds, subset)
        X_t       = torch.tensor(X_img)
        with torch.no_grad():
            recon_t = model(X_t)[0].numpy()

        classes   = np.unique(y_orig)
        n_classes = len(classes)
        shape     = SHAPE[ds]

        fig, axes = plt.subplots(2, n_classes, figsize=(1.8 * n_classes, 4))
        fig.suptitle(f'{label} — Original (top) vs Reconstruction (bottom), k={k_star}',
                     fontsize=11, fontweight='bold')

        for col_i, cls in enumerate(classes):
            sample_idx = np.where(y_orig == cls)[0][0]
            orig  = np.clip((X_img[sample_idx].reshape(shape) + 1) / 2, 0, 1)
            recon = np.clip((recon_t[sample_idx].reshape(shape) + 1) / 2, 0, 1)
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
        plt.close()

    # -------------------------------------------------------------------------
    # 6. [paul15] Archetype mixing weight distributions per cell type
    # -------------------------------------------------------------------------

    if ds == 'paul15':
        if not HAS_SEABORN:
            print('Skipping mixing weight plot — seaborn not installed.')
        else:
            model_mw = rebuild_model(result)
            X_paul15, _, label_names_mw = reload_paul15(subset)
            X_t_mw = torch.tensor(X_paul15)
            with torch.no_grad():
                z_mw    = model_mw.encode(X_t_mw)
                weights = torch.softmax(z_mw, dim=1).numpy()   # (N, k)

            k_mw       = weights.shape[1]
            cell_types  = sorted(set(label_names_mw))

            rows = [{'cell_type': label_names_mw[n],
                     'archetype': f'A{arc + 1}',
                     'weight':    float(weights[n, arc])}
                    for n in range(len(weights))
                    for arc in range(k_mw)]
            df = pd.DataFrame(rows)

            n_cols    = min(5, k_mw)
            n_rows    = int(np.ceil(k_mw / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(4.5 * n_cols, 3.5 * n_rows),
                                     sharey=True)
            fig.suptitle(f'{label} — Archetype Mixing Weights per Cell Type',
                         fontsize=13, fontweight='bold')
            axes_flat = np.array(axes).ravel()

            for arc_i in range(k_mw):
                ax     = axes_flat[arc_i]
                arc_df = df[df['archetype'] == f'A{arc_i + 1}']
                sns.violinplot(data=arc_df, x='cell_type', y='weight', ax=ax,
                               order=cell_types, inner='box', color='steelblue', cut=0)
                ax.set_title(f'A{arc_i + 1}', fontsize=10, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Weight' if arc_i % n_cols == 0 else '')
                ax.set_xticklabels(cell_types, rotation=45, ha='right', fontsize=7)
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, axis='y', alpha=0.3)

            for i in range(k_mw, len(axes_flat)):
                axes_flat[i].set_visible(False)

            plt.tight_layout()
            path = os.path.join(args.outdir, f'{prefix}_mixing_weights.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f'Saved: {path}')
            plt.close()

    print(f'\nAll figures saved to {args.outdir}/')


if __name__ == '__main__':
    main()
