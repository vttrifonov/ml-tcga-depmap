import pandas as pd
import xarray as xa
import numpy as np
import dask_ml.preprocessing as dmlp
from types import SimpleNamespace
import dask.array as daa
from joblib import Parallel, delayed
from depmap_crispr import crispr as depmap_crispr
from depmap_expr import expr as depmap_expr
from depmap_cnv import cnv as depmap_cnv
from gdc_expr import expr as gdc_expr
from common.defs import lazy_property

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

def _score3(x8_4, x8_5, n1, n2):
    x8_1 = SVD.from_data(x8_4).cut(n1).inv
    x8_2 = SVD.from_data(x8_5).cut(n2)
    x8_3 = SVD.from_data(x8_1.vs.T @ x8_2.us)
    x8_3.u = x8_1.u @ x8_3.u
    x8_3.v = x8_2.v @ x8_3.v
    return x8_3

class model:
    def __init__(self, train_split, dims):
        self.train_split = train_split
        self.dims = dims

        split = x4.dm_expr.rows
        split['train'] = ('rows', np.random.random(split.rows.shape)<train_split)

        self.train = [
            x4.dm_expr.sel(rows=split.train).data,
            x4.crispr.sel(rows=split.train).data,
            split.rows[split.train]
        ]

        self.test = [
            x4.dm_expr.sel(rows=~split.train).data,
            x4.crispr.sel(rows=~split.train).data,
            split.rows[~split.train]
        ]

        fit = _score3(self.train[0], self.train[1], np.s_[:dims], np.s_[:]).cut(np.s_[:])
        self.fit = [
            fit.mult(self.train[0]),
            fit.mult(self.test[0]),
            fit.mult(x4.gdc_expr.data)
        ]

        perm = _perm(self.train[0])
        perm_fit = _score3(perm, self.train[1], np.s_[:dims], np.s_[:]).cut(np.s_[:]).mult(perm)

        self._stats = [
            ((self.train[1]-self.fit[0])**2).mean(axis=0),
            ((self.test[1]-self.fit[1])**2).mean(axis=0),
            ((self.train[1] - perm_fit) ** 2).mean(axis=0)
        ]

    @lazy_property
    def stats(self):
        stats = Parallel(n_jobs=3)(delayed(lambda x: x.compute())(x) for x in self._stats)
        stats = pd.DataFrame(dict(
            crispr_cols=x4.crispr.crispr_cols.values,
            train=stats[0].ravel(),
            test=stats[1].ravel(),
            rand=stats[2].ravel()
        ))
        return stats

    def data(self, idx):
        data = [
            self.fit[0][:,idx], self.train[1][:,idx],
            self.fit[1][:,idx], self.test[1][:,idx],
            self.fit[2][:,idx]

        ]
        data = Parallel(n_jobs=4)(delayed(lambda x: x.compute())(x) for x in data)
        data = [
            pd.DataFrame(dict(
                pred = data[0],
                obs = data[1],
                CCLE_Name = x4.x1.dm_expr.CCLE_Name.loc[self.train[2]].values
            )),
            pd.DataFrame(dict(
                pred = data[2],
                obs = data[3],
                CCLE_Name = x4.x1.dm_expr.CCLE_Name.loc[self.test[2]].values
            )),
            pd.DataFrame(dict(
                expr=data[4],
                project_id=x4.x1.gdc_expr.project_id.values,
                is_normal=x4.x1.gdc_expr.is_normal.values
            )),
        ]
        return data

x1 = model(0.8, 400)

x1.stats.sort_values('train')

x2 = x1.data(6665)