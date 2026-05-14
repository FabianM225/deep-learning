import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.spatial.distance import pdist, squareform

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def preprocess(X):
    """Mean-centers the data and computes the total variance (mSST)."""
    meanX = np.mean(X, axis=0)
    X_centered = X - meanX
    mSST = np.sum(np.mean(X_centered**2, axis=0))
    return X_centered, mSST


def ArchetypeConsistency(XC1, XC2, mSST):
    """Calculates consistency and ISI between two sets of archetypes."""
    D = squareform(pdist(np.hstack((XC1, XC2)).T, 'euclidean'))**2
    D = D[:XC1.shape[1], XC1.shape[1]:]

    i, j, v = [], [], []
    K = XC1.shape[1]
    for k in range(K):
        min_index = np.unravel_index(np.argmin(D, axis=None), D.shape)
        i.append(min_index[0])
        j.append(min_index[1])
        v.append(D[i[-1], j[-1]])
        D[i[-1], :] = np.inf
        D[:, j[-1]] = np.inf

    consistency = 1 - np.mean(v) / mSST

    D2 = np.abs(np.corrcoef(np.hstack((XC1, XC2)).T))
    D2 = D2[:K, K:]
    ISI = 1 / (2 * K * (K - 1)) * (
        np.sum(D2 / np.max(D2, axis=1, keepdims=True)
               + D2 / np.max(D2, axis=0, keepdims=True)) - 2 * K
    )
    return consistency, ISI


def calcMI(z1, z2):
    eps = 10e-16
    P = z1 @ z2.T
    PXY = P / P.sum()
    PXPY = np.outer(np.expand_dims(PXY.sum(1), axis=0),
                    np.expand_dims(PXY.sum(0), axis=1))
    MI = np.sum(PXY * np.log(eps + PXY / (eps + PXPY)))
    return MI


def calcMI_torch(z1, z2, eps=1e-12):
    P = z1 @ z2.t()
    PXY = P / (P.sum() + eps)
    PX = PXY.sum(dim=1, keepdim=True)
    PY = PXY.sum(dim=0, keepdim=True)
    PXPY = PX @ PY
    MI = (PXY * torch.log((PXY + eps) / (PXPY + eps))).sum()
    return MI


def calcNMI(z1, z2):
    return (2 * calcMI(z1, z2)) / (calcMI(z1, z1) + calcMI(z2, z2))


def calcNMI_torch(z1, z2, eps=1e-12):
    mi12 = calcMI_torch(z1, z2)
    mi11 = calcMI_torch(z1, z1)
    mi22 = calcMI_torch(z2, z2)
    return (2 * mi12) / (mi11 + mi22 + eps)


def plot_loss_arc(L, n_arc_list, i, color, model, dataset, savedir=None):
    fig, ax = plt.subplots(figsize=(15, 5), layout='constrained')
    ax.errorbar(n_arc_list, np.mean(L, axis=1), yerr=np.std(L, axis=1),
                c=color, label=f'{model} - {dataset}')
    ax.set_xticks(n_arc_list)
    ax.tick_params(axis='x')
    ax.set_xlabel('Number of Archetypes', fontsize=30)
    ax.set_ylabel('Loss', fontsize=30)
    plt.legend(fontsize=30)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    if savedir is not None:
        plt.savefig(savedir + f"/loss_layer_" + str(i) + ".png")
        plt.close()
    else:
        plt.show()


def plot_nmi_stability(NMI, n_arc_list, model, dataset, colors, savedir=None):
    df = pd.DataFrame(NMI.T)
    df['Method'] = f'Model {model}'
    df = df.melt(id_vars='Method', var_name='Archetypes', value_name='NMI')
    fig, ax = plt.subplots(1, 1, figsize=(15, 5), layout='constrained')
    ax = sns.boxplot(x='Archetypes', y='NMI', hue='Method', showmeans=True,
                     data=df, palette=colors,
                     meanprops={"marker": "s", "markerfacecolor": "white",
                                "markeredgecolor": "black"})
    ax.xaxis.grid(True, which='major')
    [ax.axvline(x + .5, color='k') for x in ax.get_xticks()]
    plt.xticks(np.arange(len(n_arc_list)), n_arc_list, fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xlabel('Number of archetypes', fontsize=30)
    ax.set_ylabel('NMI', fontsize=30)
    ax.set_ylim([0, 1.05])
    plt.legend(fontsize=30, loc='lower right')
    plt.title(f'{model} - {dataset}', fontsize=30)
    if savedir is not None:
        plt.savefig(savedir + f"/NMI_model_" + str(model) + "dataset_" + str(dataset) + ".png")
        plt.close()
    else:
        plt.show()
