import importlib
import pandas as pd
import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklp
import sklearn.decomposition as skld
from sklearn.pipeline import Pipeline
from types import SimpleNamespace
from common.defs import lazy_property
import tensorflow as tf
import tensorflow.keras as tfk

import depmap_crispr
import depmap_cnv
importlib.reload(depmap_crispr)
importlib.reload(depmap_cnv)
from depmap_crispr import crispr as depmap_crispr
from depmap_cnv import cnv as depmap_cnv

x1 = depmap_crispr.data
x1['var'] = x1.crispr.var(axis=0)
x1['mean'] = x1.crispr.mean(axis=0)
x1['nans'] = np.isnan(x1.crispr).sum(axis=0)

#plt.scatter(
#    x1['mean'],
#    np.log10(x1['var']+1e-10)
#)
#plt.gca().axvline(x=1.5)

x2 = depmap_cnv.data
x2['var'] = x2.cnv.var(axis=0)
x2['mean'] = x2.cnv.mean(axis=0)
x2['nans'] = np.isnan(x2.cnv).sum(axis=0)

#plt.scatter(
#    x2['mean'],
#    x2['var']
#)

x4 = x2.sel(cols=x2.nans==0)
x4 = x4.sel(rows=list(set(x4.rows.values).intersection(x1.rows.values)))
x4_1 = x1.sel(cols=x1.nans==0).rename({'cols': 'crispr_cols'})
x4['crispr'] = x4_1.crispr
x4['crispr_symbol'] = x4_1.symbol

x4.cnv.values = sklp.StandardScaler().fit_transform(x4.cnv.values)
x4.crispr.values = sklp.StandardScaler().fit_transform(x4.crispr.values)

x10_1 = list(set(x4.cols.values).intersection(x4.crispr_cols.values))
x10_2 = x4.crispr.loc[:,x10_1].values
x10_3 = x4.cnv.loc[:,x10_1].values
x10_4 = (x10_2*x10_3).mean(axis=0)

plt.plot(x10_4, '.')


x6_1 = np.linalg.svd(x4.crispr.values)
x6_2 = np.linalg.svd(x4.cnv.values)

def _score(permute=False):
    x6_3_1 = x6_1[0]
    if permute:
        x6_3_1 = x6_3_1[np.random.permutation(x6_3_1.shape[0]), :]
    x6_3 = x6_2[0].T @ x6_3_1
    x6_3 = x6_3 ** 2
    x6_3 = np.apply_along_axis(np.cumsum, 1, x6_3)
    x6_3 = np.apply_along_axis(np.cumsum, 0, x6_3)
    x6_4 = np.full_like(x6_3, 1 / len(x6_1[1]))
    x6_4 = np.apply_along_axis(np.cumsum, 1, x6_4)
    x6_4 = np.apply_along_axis(np.cumsum, 0, x6_4)
    x6_3 /= x6_4
    return x6_3

x7_1 = _score(False)

x7_2 = _score(True)
plt.plot(sorted(x7_1.max(axis=1)), sorted(x7_2.max(axis=1)), '.')

x8_1 = [_score(True).max() for _ in range(100)]
(x8_1>x7_1.max()).mean()

plt.imshow(x7_1[:100,:100])

def _score1(permute=False):
    x6_3_1 = x6_1[0]
    if permute:
        x6_3_1 = x6_3_1[np.random.permutation(x6_3_1.shape[0]), :]
    x6_3 = x4.cnv.values.T @ x6_3_1
    x6_3 = x6_3 ** 2
    x6_3 = np.apply_along_axis(np.cumsum, 1, x6_3)
    x6_3 /= x6_3[:,-1:]
    x6_4 = np.full_like(x6_3, 1 / len(x6_1[1]))
    x6_4 = np.apply_along_axis(np.cumsum, 1, x6_4)
    x6_3 /= x6_4
    return x6_3

x9_1 = _score1(False)

x9_2 = _score1(True)
plt.plot(sorted(x9_1.max(axis=1)), sorted(x9_2.max(axis=1)), '.')
plt.gca().axline((0,0), slope=1)

plt.plot(x9_1[np.argsort(x9_1.max(axis=1))[-1],:], '.')

plt.imshow(x9_1>10, aspect='auto')