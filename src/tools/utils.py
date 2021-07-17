import json

import numpy as np
import pandas as pd
from numba import jit, njit, prange
from sklearn.preprocessing import scale


def load_clades(clades_path="data/clades.json", size=30):
    with open(clades_path, "r") as f:
        clades = json.load(f)

    clade_dict = {}
    clade_sizes = {}
    for k, v in clades.items():
        if v["size"] > size:
            clade_dict[k] = v["inds"]
            clade_sizes[k] = v["size"]

    clade_sizes = pd.Series(clade_sizes).sort_values(ascending=False)
    return ((clade_dict, clade_sizes))


def load_NPP_mat(path="data/NPP.tsv", scale_mat=False):
    '''
    Loads the NPP and turns it into a matrix
    :param path: path to load from
    :return: a numpy 19891x881 matrix - rows are genes, columns are species and the list of genes
    '''

    f = open(path, 'r')
    NPP = pd.read_table(f, delimiter="\t", index_col=0)

    if "9606" in NPP.columns:
        NPP = NPP.drop(columns="9606")

    NPP_genes = {x: i for i, x in enumerate(NPP.index)}
    NPP = NPP.values

    if scale_mat:
        NPP = scale(NPP, axis=0, with_mean=True, with_std=True)
        # NPP = scale(NPP, axis=1, with_mean=True, with_std=True)
    print("NPP loaded")

    return (NPP, NPP_genes)


@njit
def cov(a, b):
    mA = np.mean(a)
    mB = np.mean(b)
    cov = ((a - mA) * (b - mB)).mean()
    return (cov)

@njit
def cor(a, b):
    mA = np.mean(a)
    mB = np.mean(b)

    numerator = np.sum((a - mA) * (b - mB)) + 1e-10
    denominator = np.sqrt(np.sum((a - mA) ** 2) * np.sum((b - mB) ** 2)) + 1e-10
    return np.divide(numerator, denominator)

@njit
def manhattan(a, b):
    return np.sum(np.abs(a - b))

@njit
def jaccard(a, b):
    sum_a = a.sum()
    sum_b = b.sum()
    inter = a.dot(b)
    if (sum_a + sum_b - inter)==0:
        return 0
    return inter / (sum_a + sum_b - inter)


@njit(parallel=True)
def calc_any(NPP, mask1, mask2, func):
    res = np.zeros(mask1.size)
    for i in prange(mask1.size):
        res[i] = func(NPP[mask1[i], :], NPP[mask2[i], :])
    return res

def XY(dat, feats, targets, ind=None):
    """
    Gets dat object (with all columns) and splits for X and y by features and targets
    :param dat: DF, has all needed information
    :param feats: columns to use as features, passed to x
    :param targets: labels for training, passed to y
    :param ind: Index object, optional, which rows to use.
    :return: Tuple of two data frames, X and y
    """
    if ind is None:
        return ((dat.loc[:, feats], dat.loc[:, targets]))
    else:
        return ((dat.loc[ind, feats], dat.loc[ind, targets]))


def print_write(s, f):
    s = str(s)
    print(s)
    f.write(s + "\n")
