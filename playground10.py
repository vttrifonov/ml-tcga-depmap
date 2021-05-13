import importlib
import pandas as pd
import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklp
import sklearn.decomposition as skld
import dask_ml.decomposition as dmld
import dask_ml.preprocessing as dmlp
from sklearn.pipeline import Pipeline
from types import SimpleNamespace
from common.defs import lazy_property
import tensorflow as tf
import tensorflow.keras as tfk
from joblib import Parallel, delayed
import dask as da
import dask.array as daa

import depmap_crispr
import depmap_expr
import gdc_expr
importlib.reload(depmap_crispr)
importlib.reload(depmap_expr)
importlib.reload(gdc_expr)
from depmap_crispr import crispr as depmap_crispr
from depmap_expr import expr as depmap_expr
from gdc_expr import expr as gdc_expr

x1 = SimpleNamespace()

x1.crispr = depmap_crispr.data.copy()
x1.crispr = x1.crispr.sel(cols=np.isnan(x1.crispr.mat).sum(axis=0)==0)
x1.crispr = x1.crispr.sel(rows=np.isnan(x1.crispr.mat).sum(axis=1)==0)
x1.crispr['mat'] = (('rows', 'cols'), daa.from_array(x1.crispr.mat.values))
x1.crispr = x1.crispr.rename({'mat': 'crispr', 'cols': 'crispr_cols'})

x1.dm_expr = depmap_expr.data.copy()
x1.dm_expr = x1.dm_expr.sel(cols=np.isnan(x1.dm_expr.mat).sum(axis=0)==0)
x1.dm_expr = x1.dm_expr.sel(rows=np.isnan(x1.dm_expr.mat).sum(axis=1)==0)
x1.dm_expr['mean'] = x1.dm_expr.mat.mean(axis=0)
x1.dm_expr = x1.dm_expr.sel(cols=x1.dm_expr['mean']>1.5)
x1.dm_expr['mat'] = (('rows', 'cols'), daa.from_array(x1.dm_expr.mat.values))
x1.dm_expr = x1.dm_expr.rename({'mat': 'dm_expr', 'cols': 'expr_cols'})

x1_1 = gdc_expr.mat2.col_entrez[['col', 'dbprimary_acc', 'display_label']]
x1_1 = x1_1.drop_duplicates()
x1_1 = x1_1.rename(columns={
    'col': 'cols',
    'dbprimary_acc': 'entrez',
    'display_label': 'symbol'
})
x1_1['expr_cols'] = x1_1.symbol + ' (' + x1_1.entrez + ')'
x1_1 = x1_1.query('expr_cols.isin(@x1.dm_expr.expr_cols.values)').copy()
x1_1['n'] = x1_1.groupby('expr_cols').expr_cols.transform('size')
x1_1 = x1_1.query('n==1 | cols.str.find("ENSGR")<0')
x1_1['n'] = x1_1.groupby('expr_cols').expr_cols.transform('size')
x1_1 = x1_1.query('n==1')
del x1_1['n']
x1_1 = x1_1.set_index('cols').to_xarray()

x1.gdc_expr = gdc_expr.mat2.xarray[['data', 'rows', 'cols']]
x1.gdc_expr = x1.gdc_expr.sel(cols=x1_1.cols)
x1.gdc_expr = x1.gdc_expr.merge(x1_1)
x1.gdc_expr = x1.gdc_expr.swap_dims({'cols': 'expr_cols'})
del x1.gdc_expr['cols']
x1.gdc_expr = x1.gdc_expr.sel(expr_cols=daa.isnan(x1.gdc_expr.data.data).sum(axis=0).compute()==0)
x1.gdc_expr = x1.gdc_expr.sel(rows=daa.isnan(x1.gdc_expr.data.data).sum(axis=1).compute()==0)
x1.gdc_expr['mean'] = x1.gdc_expr.data.mean(axis=0).compute()
x1.gdc_expr = x1.gdc_expr.sel(expr_cols=x1.gdc_expr['mean']>(-7))
x1.gdc_expr = x1.gdc_expr.rename({'data': 'gdc_expr', 'rows': 'gdc_expr_rows'})

x4_1 = set(x1.crispr.rows.values)
x4_1.intersection_update(x1.dm_expr.rows.values)
x4_1 = list(x4_1)

x4_3 = x1.gdc_expr.expr_cols.values
x4_3 = pd.Series(range(len(x4_3)), index=x4_3)

x4_2 = set(x1.dm_expr.expr_cols.values)
x4_2.intersection_update(x1.gdc_expr.expr_cols.values)
x4_2 = list(x4_2)
x4_2 = x4_3[x4_2].sort_values()
x4_2 = list(x4_2.index)

x4 = xa.merge([
    x1.crispr.crispr.loc[x4_1,:].astype('float32'),
    x1.dm_expr.dm_expr.loc[x4_1, x4_2].astype('float32'),
    x1.gdc_expr.gdc_expr.loc[:, x4_2].astype('float32')
])
x4.dm_expr.data = dmlp.StandardScaler().fit_transform(x4.dm_expr.data)
x4.gdc_expr.data = dmlp.StandardScaler().fit_transform(x4.gdc_expr.data)
x4.crispr.data = dmlp.StandardScaler().fit_transform(x4.crispr.data)

def _svd_u(x):
    pca = dmld.PCA(n_components=x.shape[0]-1).fit(x)
    u = pca.transform(x) / (pca.singular_values_.reshape(1,-1))
    return SimpleNamespace(u=u, s=pca.singular_values_)

x6_1 = [x4.crispr.data, x4.dm_expr.data, x4.gdc_expr.data]
x6 = Parallel(n_jobs=4)(delayed(_svd_u)(v) for v in x6_1)
#for i, x in enumerate(x6):
#    x.v = (x6_1[i].T @ x.u) / (x.s.reshape(1,-1))

def _score(x6_1, x6_2, permute=False):
    if permute:
        x6_2 = x6_2[np.random.permutation(x6_2.shape[0]), :]
    x6_3 = x6_1.T @ x6_2
    x6_3 = x6_3 ** 2
    x6_3 = np.apply_along_axis(np.cumsum, 1, x6_3)
    x6_3 = np.apply_along_axis(np.cumsum, 0, x6_3)
    x6_4 = np.full_like(x6_3, 1 / x6_1.shape[0])
    x6_4 = np.apply_along_axis(np.cumsum, 1, x6_4)
    x6_4 = np.apply_along_axis(np.cumsum, 0, x6_4)
    x6_3 /= x6_4
    return x6_3

x7_1 = _score(x6[2].u, x6[0].u, permute=False)

pd.Series(np.round(x7_1.ravel(), 1)).value_counts().sort_index()[:3]

plt.imshow(np.where(x7_1>1.5, 1.5, np.where(x7_1<1.1, 0, np.round(x7_1, 1))))

def _score1(x6_1, x6_2, permute=False):
    if permute:
        x6_2 = x6_2[np.random.permutation(x6_2.shape[0]), :]
    x6_3 = x6_1.T @ x6_2
    x6_3 = x6_3 ** 2
    x6_3 = np.apply_along_axis(np.cumsum, 1, x6_3)
    x6_3 /= x6_3[:,-1:]
    x6_4 = np.full_like(x6_3, 1 / x6_1.shape[0])
    x6_4 = np.apply_along_axis(np.cumsum, 1, x6_4)
    x6_3 /= x6_4
    return x6_3

x8_1 = _score1(x4.crispr.values, x6[2].u)
plt.plot(sorted(x8_1.max(axis=1).ravel()), '.')

x8_2 = _score1(x4.crispr.values, x6[2].u, permute=True)
plt.plot(sorted(x8_2.max(axis=1).ravel()), '.')

(x8_1.max(axis=1)>16).sum()

plt.plot(pd.Series(np.argmax(x8_1, axis=1)).value_counts().sort_index(), '.')

plt.plot(x8_1.mean(axis=0), '.')

