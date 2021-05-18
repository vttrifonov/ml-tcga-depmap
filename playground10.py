import importlib
import pandas as pd
import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
import depmap_cnv
import gdc_expr
importlib.reload(depmap_crispr)
importlib.reload(depmap_expr)
importlib.reload(depmap_cnv)
importlib.reload(gdc_expr)
from depmap_crispr import crispr as depmap_crispr
from depmap_expr import expr as depmap_expr
from depmap_cnv import cnv as depmap_cnv
from gdc_expr import expr as gdc_expr

class x4:
    x1 = SimpleNamespace()

    x1.crispr = depmap_crispr.data.copy()
    x1.crispr = x1.crispr.sel(cols=np.isnan(x1.crispr.mat).sum(axis=0)==0)
    x1.crispr = x1.crispr.sel(rows=np.isnan(x1.crispr.mat).sum(axis=1)==0)
    x1.crispr['mat'] = (('rows', 'cols'), daa.from_array(x1.crispr.mat.values))
    x1.crispr = x1.crispr.rename({'mat': 'crispr', 'cols': 'crispr_cols'})

    x1.dm_expr = depmap_expr.data.copy()
    x1.dm_expr = x1.dm_expr.merge(
        depmap_expr.release.samples.rename(columns={'DepMap_ID': 'rows'}).set_index('rows').to_xarray(),
        join='inner'
    )
    x1.dm_expr = x1.dm_expr.sel(cols=np.isnan(x1.dm_expr.mat).sum(axis=0)==0)
    x1.dm_expr = x1.dm_expr.sel(rows=np.isnan(x1.dm_expr.mat).sum(axis=1)==0)
    x1.dm_expr['mean'] = x1.dm_expr.mat.mean(axis=0)
    x1.dm_expr = x1.dm_expr.sel(cols=x1.dm_expr['mean']>1.5)
    x1.dm_expr['mat'] = (('rows', 'cols'), daa.from_array(x1.dm_expr.mat.values))
    x1.dm_expr = x1.dm_expr.rename({'mat': 'dm_expr', 'cols': 'expr_cols'})

    x1.dm_cnv = depmap_cnv.data.copy()
    x1.dm_cnv = x1.dm_cnv.sel(cols=np.isnan(x1.dm_cnv.mat).sum(axis=0)==0)
    x1.dm_cnv = x1.dm_cnv.sel(rows=np.isnan(x1.dm_cnv.mat).sum(axis=1)==0)
    x1.dm_cnv['mat'] = (('rows', 'cols'), daa.from_array(x1.dm_cnv.mat.values))
    x1.dm_cnv = x1.dm_cnv.rename({'mat': 'dm_cnv', 'cols': 'cnv_cols'})

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

    x1.gdc_expr = gdc_expr.mat2.xarray[['data', 'rows', 'cols', 'project_id', 'is_normal']]
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
    x4_1.intersection_update(x1.dm_cnv.rows.values)
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
        x1.dm_cnv.dm_cnv.loc[x4_1,:].astype('float32'),
        x1.dm_expr.dm_expr.loc[x4_1, x4_2].astype('float32'),
        x1.gdc_expr.gdc_expr.loc[:, x4_2].astype('float32')
    ])
    x4.crispr.data = dmlp.StandardScaler().fit_transform(x4.crispr.data)
    x4.dm_expr.data = dmlp.StandardScaler().fit_transform(x4.dm_expr.data)
    x4.dm_cnv.data = dmlp.StandardScaler().fit_transform(x4.dm_cnv.data)
    x4.gdc_expr.data = dmlp.StandardScaler().fit_transform(x4.gdc_expr.data)

    crispr = x4.crispr
    dm_expr = x4.dm_expr
    dm_cnv = x4.dm_cnv
    gdc_expr = x4.gdc_expr

def _perm(x):
    return x[np.random.permutation(x.shape[0]), :]

class SVD:
    def __init__(self, u, s, v):
        self.u = u
        self.s = s
        self.v = v

    @staticmethod
    def from_data(data, n = None, solver = 'full'):
        if n is None:
            n = min(*data.shape)

        if solver == 'full':
            svd = daa.linalg.svd(data)
        elif solver == 'rand':
            svd = daa.linalg.svd_compressed(data, n)
        else:
            raise ValueError('unknown solver')

        return SVD(svd[0][:,:n], svd[1][:n], svd[2][:n,:].T)

    def cut(self, n=None):
        if n is None:
            n = np.s_[:]
        return SVD(self.u[:, n], self.s[n], self.v[:, n])

    @property
    def us(self):
        return self.u * self.s.reshape(1, -1)

    @property
    def vs(self):
        return self.v * self.s.reshape(1, -1)

    @property
    def usv(self):
        return self.us @ self.v.T

    @property
    def perm(self):
        return SVD(_perm(self.u), self.s, self.v)

    @property
    def inv(self):
        return SVD(self.v, 1/self.s, self.u)

    @property
    def T(self):
        return SVD(self.v, self.s, self.u)

    def mult(self, x):
        return (x @ self.us) @ self.v.T

class x6:
    crispr = SVD.from_data(x4.crispr.data, x4.crispr.shape[0]-1)
    dm_expr = SVD.from_data(x4.dm_expr.data, x4.dm_expr.shape[0]-1)
    dm_cnv = SVD.from_data(x4.dm_cnv.data, x4.dm_cnv.shape[0]-1)
    #x6.gdc_expr = SVD.from_data(x4.gdc_expr.data, 5000, 'rand')

def _score1(x6_1, x6_2):
    x6_3 = x6_1.T @ x6_2
    x6_3 = x6_3 ** 2
    x6_3 = np.apply_along_axis(np.cumsum, 1, x6_3)
    return x6_3

def _score2(x6_1, x6_2):
    x6_3 = _score1(x6_1, x6_2)
    x6_3 = np.apply_along_axis(np.cumsum, 0, x6_3)
    return x6_3

def _score(x6_1, x6_2, scorer):
    return scorer(x6_1, x6_2)/scorer(x6_1, _perm(x6_2))

plt.imshow(_score((x6.crispr.u), x6.dm_expr.u, _score2), aspect='auto')
plt.clim(0, 3)
plt.colorbar()

plt.imshow(_score((x6.dm_expr.u), x6.dm_cnv.u, _score2), aspect='auto')
plt.clim(0, 3)
plt.colorbar()

plt.imshow(_score((x6.gdc_expr.v), x6.dm_expr.v, _score2), aspect='auto')
plt.clim(0, 8)
plt.colorbar()

def _score3(x8_4, x8_5, n1, n2):
    x8_1 = SVD.from_data(x8_4).cut(n1).inv
    x8_2 = SVD.from_data(x8_5).cut(n2)
    x8_3 = SVD.from_data(x8_1.vs.T @ x8_2.us)
    x8_3.u = x8_1.u @ x8_3.u
    x8_3.v = x8_2.v @ x8_3.v
    return x8_3

x11_10 = x4.dm_expr.rows
x11_10['train'] = ('rows', np.random.random(x11_10.rows.shape)<0.8)

x11_6 = (x4.dm_expr.sel(rows=x11_10.train).data, x4.crispr.sel(rows=x11_10.train).data, np.s_[:400], np.s_[:], np.s_[:])
x11_8 = _score3(*x11_6[:4])
x11_1 = x11_8.cut(x11_6[4]).mult(x11_6[0])
x11_3 = (x11_6[1]-x11_1).compute()

x11_11 = x11_8.cut(x11_6[4]).mult(x4.dm_expr.sel(rows=~x11_10.train).data)
x11_13 = (x4.crispr.sel(rows=~x11_10.train).data-x11_11).compute()

x11_7 = _perm(x11_6[0])
x11_2 = _score3(x11_7, *x11_6[1:4]).cut(x11_6[4]).mult(x11_7)
x11_4 = (x11_6[1]-x11_2).compute()
x11_5 = pd.DataFrame(dict(
    crispr_cols=x4.crispr.crispr_cols.values,
    obs=np.mean(x11_3**2, axis=0).ravel(),
    pred=np.mean(x11_13**2, axis=0).ravel(),
    rand=np.mean(x11_4**2, axis=0).ravel(),
))

plt.figure().gca().plot(sorted(x11_5.obs), sorted(x11_5.rand), '.')
plt.gcf().gca().axline(tuple([x11_5[['obs', 'rand']].min().min()]*2), slope=1)

plt.figure().gca().hist2d(x11_5.obs, x11_5.pred, bins=200)

x11_5.sort_values('pred').head(20)

plt.plot(x4.crispr.sel(rows=x11_10.train).data[:,14447], x11_1[:,14447], '.')
plt.plot(x4.crispr.sel(rows=~x11_10.train).data[:,14447], x11_11[:,14447], '.')

x11_5.sort_values('obs').head(20)
x11_5.sort_values('obs').query('obs<0.2')

pd.DataFrame(dict(obs=x11_5.obs<0.3, pred=x11_5.pred<0.73)).value_counts().\
    rename('count').reset_index().\
    pivot('obs', 'pred', 'count')

x11_9 = pd.DataFrame(dict(
    expr = x11_1[:,8033].compute(),
    CCLE_Name = x4.x1.dm_expr.CCLE_Name.loc[x4.dm_expr.rows].values
))
px.scatter(
    x11_9.reset_index(),
    x='index', y='expr',
    color='CCLE_Name'
).show()

x12_1 = x11_8.cut(x11_6[4]).mult(x4.gdc_expr.data)
x12_2 = pd.DataFrame(dict(
    expr = x12_1[:,14447].compute(),
    project_id = x4.x1.gdc_expr.project_id.values,
    is_normal = x4.x1.gdc_expr.is_normal.values
))
px.scatter(
    x12_2.reset_index(),
    x='index', y='expr',
    color='project_id', symbol='is_normal'
).show()
