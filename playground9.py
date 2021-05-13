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
from joblib import Parallel, delayed

import depmap_crispr
import depmap_cnv
import depmap_expr
importlib.reload(depmap_crispr)
importlib.reload(depmap_cnv)
importlib.reload(depmap_expr)
from depmap_crispr import crispr as depmap_crispr
from depmap_cnv import cnv as depmap_cnv
from depmap_expr import expr as depmap_expr

x1 = SimpleNamespace()

x1.crispr = depmap_crispr.data.copy()
x1.crispr = x1.crispr.sel(cols=np.isnan(x1.crispr.mat).sum(axis=0)==0)
x1.crispr = x1.crispr.sel(rows=np.isnan(x1.crispr.mat).sum(axis=1)==0)
x1.crispr = x1.crispr.rename({'mat': 'crispr', 'cols': 'crispr_cols'})

x1.cnv = depmap_cnv.data.copy()
x1.cnv = x1.cnv.sel(cols=np.isnan(x1.cnv.mat).sum(axis=0)==0)
x1.cnv = x1.cnv.sel(rows=np.isnan(x1.cnv.mat).sum(axis=1)==0)
x1.cnv = x1.cnv.rename({'mat': 'cnv', 'cols': 'cnv_cols'})

x1.expr = depmap_expr.data.copy()
x1.expr = x1.expr.sel(cols=np.isnan(x1.expr.mat).sum(axis=0)==0)
x1.expr = x1.expr.sel(rows=np.isnan(x1.expr.mat).sum(axis=1)==0)
x1.expr['mean'] = x1.expr.mat.mean(axis=0)
x1.expr = x1.expr.sel(cols=x1.expr['mean']>1.5)
x1.expr = x1.expr.rename({'mat': 'expr', 'cols': 'expr_cols'})

x4_1 = set(x1.crispr.rows.values)
x4_1.intersection_update(x1.cnv.rows.values)
x4_1.intersection_update(x1.expr.rows.values)
x4_1 = list(x4_1)

x4 = xa.merge([
    x1.crispr.crispr.loc[x4_1,:].astype('float32'),
    x1.cnv.cnv.loc[x4_1,:].astype('float32'),
    x1.expr.expr.loc[x4_1,:].astype('float32')
])
x4.expr.values = sklp.StandardScaler().fit_transform(x4.expr.values)
x4.cnv.values = sklp.StandardScaler().fit_transform(x4.cnv.values)
x4.crispr.values = sklp.StandardScaler().fit_transform(x4.crispr.values)

x5_1 = set(x4.cnv_cols.values)
x5_1.intersection_update(x4.expr_cols.values)
x5_1 = np.array(list(x5_1))
x5_2 = x4.expr.loc[:,x5_1].values
x5_3 = x4.cnv.loc[:,x5_1].values

plt.plot((x5_2*x5_3).mean(axis=0), '.')

plt.plot((x5_2[:,[0]] * x5_3).mean(axis=0), '.')

x5_5 = x5_1=='TP53 (7157)'
x5_5 = pd.DataFrame(dict(
    expr=x5_2[:, x5_5].ravel(),
    cnv=x5_3[:, x5_5].ravel()
))
np.corrcoef(x5_5.T)
plt.plot(x5_5.cnv, (x5_5.expr), '.')

def _svd_u(x):
    pca = skld.PCA(n_components=x.shape[0]-1).fit(x)
    u = pca.transform(x) / (pca.singular_values_.reshape(1,-1))
    return SimpleNamespace(u=u, s=pca.singular_values_)

x6_1 = [x4.crispr.values, x4.cnv.values, x4.expr.values, np.hstack([x4.cnv.values, x4.expr.values])]
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

x9_1 = x4.crispr.values
x9_2 = x6[2].u[:,:400]
#x9_2 = x9_2[np.random.permutation(x9_2.shape[0]),:]
x9_3 = x9_2 @ (x9_2.T @ x9_1)
print(((x9_1-x9_3)**2).mean())


x9_1 = x4.crispr.values
x9_2 = x6[2].u
x9_3 = x9_2.T @ x9_1
x9_3 = np.apply_along_axis(np.cumsum, 0, x9_3**2)
x9_3 /= x9_3[-1:,:]
x9_4 = x9_2[np.random.permutation(x9_2.shape[0]),:]
x9_5 = x9_4.T @ x9_1
x9_5 = np.apply_along_axis(np.cumsum, 0, x9_5**2)
x9_5 /= x9_5[-1:,:]

x9_6 = pd.DataFrame(dict(obs=x9_3[50,:], rand=x9_5[50,:]))
plt.plot(sorted(x9_6.obs), sorted(x9_6.rand), '.')
plt.gca().axline((x9_6.min().min(), x9_6.min().min()), slope=1)

x10_1 = x6[2].u[:,:100]
x10_2 = x4.expr.values.T @ x10_1
x10_3 = x4.crispr.values.T @ x10_1
x10_4 = np.linalg.qr(x10_2)
x10_3 = x10_3

