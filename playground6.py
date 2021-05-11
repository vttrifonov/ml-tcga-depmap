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

import depmap_expr
import depmap_crispr
import gdc_expr
importlib.reload(depmap_expr)
importlib.reload(depmap_crispr)
importlib.reload(gdc_expr)
from depmap_expr import expr as depmap_expr
from depmap_crispr import crispr as depmap_crispr
from gdc_expr import expr as gdc_expr

x1 = depmap_expr.data

x1['var'] = x1.mat.var(axis=0)
x1['mean'] = x1.mat.mean(axis=0)

plt.scatter(
    x1['mean'],
    np.log10(x1['var']+1e-10)
)
plt.gca().axvline(x=1.5)

data = x1.sel(cols=x1['mean']>1.5)
data['train'] = ('rows', np.random.random(data.rows.shape)<0.8)

data = SimpleNamespace(
    train=data.sel(rows=data.train).mat.values,
    test=data.sel(rows=~data.train).mat.values,
)
scaler = sklp.StandardScaler().fit(data.train)
data = SimpleNamespace(
    train=scaler.transform(data.train),
    test=scaler.transform(data.test)
)

pca = skld.PCA(n_components=data.train.shape[0]).fit(data.train)
transformed = SimpleNamespace(
    train=pca.transform(data.train),
    test=pca.transform(data.test)
)
transformed = SimpleNamespace(
    train=np.apply_along_axis(np.cumsum, 1, transformed.train**2),
    test=np.apply_along_axis(np.cumsum, 1, transformed.test**2)
)
transformed = SimpleNamespace(
    train=transformed.train.sum(axis=0)/np.prod(data.train.shape),
    test=transformed.test.sum(axis=0)/np.prod(data.test.shape)
)
plt.plot(transformed.train, transformed.test, '.')
plt.gca().axline((0,0), slope=1)



x2 = depmap_crispr.data

x2['var'] = x2.mat.var(axis=0)
x2['mean'] = x2.mat.mean(axis=0)
x2['nans'] = np.isnan(x2.mat).sum(axis=0)

plt.scatter(
    x2['mean'],
    x2['var']
)
plt.gca().axvline(x=1.5)

x4 = x2.sel(cols=x2.nans==0)
x4 = x4.sel(rows=list(set(x4.rows.values).intersection(x1.rows.values)))
x4['expr'] = x1.sel(cols=x1['mean']>1.5).mat.rename({'cols': 'expr_cols'})

x4.mat.values = sklp.StandardScaler().fit_transform(x4.mat.values)
x4.expr.values = sklp.StandardScaler().fit_transform(x4.expr.values)

x4_1 = skld.PCA(n_components=100).fit_transform(x4.expr.values)
x4_1 = x4_1[np.random.permutation(x4_1.shape[0]),:]
x4['expr_pca'] = (('rows', 'expr_pca_cols'), x4_1)

x5 = tfk.Sequential([
    tfk.layers.InputLayer(x4.expr_pca.shape[1]),
    tfk.layers.Dense(x4.mat.shape[1])
])
x5.compile(optimizer='adam', loss='mse')

x5.fit(x4.expr_pca.values, x4.mat.values, batch_size=100, epochs=100)

x6_1 = np.linalg.svd(x4.expr.values)
x6_2 = np.linalg.svd(x4.mat.values)

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
plt.plot(sorted(x7_1.ravel()), sorted(x7_2.ravel()), '.')
plt.gca().axline((0,0), slope=1)
