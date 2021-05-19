import pandas as pd
import xarray as xa
import numpy as np
import dask_ml.preprocessing as dmlp
from types import SimpleNamespace
import dask.array as daa
from depmap_crispr import crispr as depmap_crispr
from depmap_expr import expr as depmap_expr
from depmap_cnv import cnv as depmap_cnv
from gdc_expr import expr as gdc_expr
from gdc_cnv import cnv as gdc_cnv
import sparse as sp
from common.defs import lazy_property

class merge:
    @lazy_property
    def _merge(self):
        class x4:
            crispr = depmap_crispr.data.copy()
            crispr = crispr.sel(cols=np.isnan(crispr.mat).sum(axis=0)==0)
            crispr = crispr.sel(rows=np.isnan(crispr.mat).sum(axis=1)==0)
            crispr['mat'] = (('rows', 'cols'), crispr.mat.data.rechunk(-1, 1000))
            crispr = crispr.rename({'mat': 'data', 'cols': 'crispr_cols', 'rows': 'depmap_rows'})

            dm_expr = depmap_expr.data.copy()
            dm_expr = dm_expr.merge(
                depmap_expr.release.samples.rename(columns={'DepMap_ID': 'rows'}).set_index('rows').to_xarray(),
                join='inner'
            )
            dm_expr = dm_expr.sel(cols=np.isnan(dm_expr.mat).sum(axis=0)==0)
            dm_expr = dm_expr.sel(rows=np.isnan(dm_expr.mat).sum(axis=1)==0)
            dm_expr['mat'] = (('rows', 'cols'), dm_expr.mat.data.rechunk(-1, 1000))
            dm_expr['mean'] = dm_expr.mat.mean(axis=0)
            dm_expr = dm_expr.sel(cols=dm_expr['mean']>1.5)
            dm_expr = dm_expr.rename({'mat': 'data', 'cols': 'expr_cols', 'rows': 'depmap_rows'})

            dm_cnv = depmap_cnv.data.copy()
            dm_cnv = dm_cnv.sel(cols=np.isnan(dm_cnv.mat).sum(axis=0)==0)
            dm_cnv = dm_cnv.sel(rows=np.isnan(dm_cnv.mat).sum(axis=1)==0)
            dm_cnv['mat'] = (('rows', 'cols'), dm_cnv.mat.data.rechunk(-1, 1000))
            dm_cnv = dm_cnv.rename({'mat': 'data', 'cols': 'cnv_cols', 'rows': 'depmap_rows'})

            x1_1 = gdc_expr.col_entrez[['col', 'dbprimary_acc', 'display_label']]
            x1_1 = x1_1.drop_duplicates()
            x1_1 = x1_1.rename(columns={
                'col': 'cols',
                'dbprimary_acc': 'entrez',
                'display_label': 'symbol'
            })
            x1_1['expr_cols'] = x1_1.symbol + ' (' + x1_1.entrez + ')'
            x1_1 = x1_1.query('expr_cols.isin(@dm_expr.expr_cols.values)').copy()
            x1_1['n'] = x1_1.groupby('expr_cols').expr_cols.transform('size')
            x1_1 = x1_1.query('n==1 | cols.str.find("ENSGR")<0')
            x1_1['n'] = x1_1.groupby('expr_cols').expr_cols.transform('size')
            x1_1 = x1_1.query('n==1')
            del x1_1['n']
            x1_1 = x1_1.set_index('cols').to_xarray()

            x5_1 = gdc_expr.xarray[['rows', 'project_id', 'is_normal', 'case_id', 'sample_id']].to_dataframe().reset_index()
            x5_1 = x5_1.rename(columns={'sample_id': 'gdc_rows'})
            x5_2 = x5_1.drop(columns=['rows']).drop_duplicates()
            x5_3 = x5_1[['rows', 'gdc_rows']].copy()
            x5_3['w'] = x5_3.groupby('gdc_rows').gdc_rows.transform('size')
            x5_3['w'] = 1/x5_3.w
            x5_3['rows'] = pd.Series(range(x5_1.shape[0]), index=x5_1.rows)[x5_3.rows.to_numpy()].to_numpy()
            x5_3['gdc_rows'] = pd.Series(range(x5_2.shape[0]), index=x5_2.gdc_rows)[x5_3.gdc_rows.to_numpy()].to_numpy()
            x5_3 = daa.from_array(
                sp.COO(x5_3[['gdc_rows', 'rows']].to_numpy().T, x5_3.w.astype('float32')),
                chunks=(1000,-1)
            )

            gdc_expr = gdc_expr.xarray[['data', 'rows', 'cols']]
            gdc_expr = gdc_expr.sel(cols=x1_1.cols)
            gdc_expr = gdc_expr.merge(x1_1)
            gdc_expr = gdc_expr.swap_dims({'cols': 'expr_cols'})
            del gdc_expr['cols']
            gdc_expr = gdc_expr.merge(x5_2.set_index('gdc_rows'))
            x5_3 = x5_3.rechunk((None, gdc_expr.data.chunks[0]))
            gdc_expr['data'] = (('gdc_rows', 'expr_cols'),  x5_3 @ gdc_expr.data.data.astype('float32'))
            del gdc_expr['rows']
            gdc_expr['mean'] = gdc_expr.data.mean(axis=0).compute()
            gdc_expr = gdc_expr.sel(expr_cols=gdc_expr['mean']>(-7))

            x1_1 = gdc_cnv.col_entrez[['col', 'dbprimary_acc', 'display_label']]
            x1_1 = x1_1.drop_duplicates()
            x1_1 = x1_1.rename(columns={
                'col': 'cols',
                'dbprimary_acc': 'entrez',
                'display_label': 'symbol'
            })
            x1_1['cnv_cols'] = x1_1.symbol + ' (' + x1_1.entrez + ')'
            x1_1 = x1_1.query('cnv_cols.isin(@dm_cnv.cnv_cols.values)').copy()
            x1_1['n'] = x1_1.groupby('cnv_cols').cnv_cols.transform('size')
            x1_1 = x1_1.query('n==1 | cols.str.find("ENSGR")<0')
            x1_1['n'] = x1_1.groupby('cnv_cols').cnv_cols.transform('size')
            x1_1 = x1_1.query('n==1')
            del x1_1['n']
            x1_1 = x1_1.set_index('cols').to_xarray()

            x5_1 = gdc_cnv.xarray[['rows', 'project_id', 'case_id', 'sample_id']].to_dataframe().reset_index()
            x5_1 = x5_1.rename(columns={'sample_id': 'gdc_rows'})
            x5_2 = x5_1.drop(columns=['rows']).drop_duplicates()
            x5_3 = x5_1[['rows', 'gdc_rows']].copy()
            x5_3['w'] = x5_3.groupby('gdc_rows').gdc_rows.transform('size')
            x5_3['w'] = 1/x5_3.w
            x5_3['rows'] = pd.Series(range(x5_1.shape[0]), index=x5_1.rows)[x5_3.rows.to_numpy()].to_numpy()
            x5_3['gdc_rows'] = pd.Series(range(x5_2.shape[0]), index=x5_2.gdc_rows)[x5_3.gdc_rows.to_numpy()].to_numpy()
            x5_3 = daa.from_array(
                sp.COO(x5_3[['gdc_rows', 'rows']].to_numpy().T, x5_3.w.astype('float32')),
                chunks=(1000,-1)
            )

            gdc_cnv = gdc_cnv.xarray[['data', 'rows', 'cols']]
            gdc_cnv = gdc_cnv.sel(cols=x1_1.cols)
            gdc_cnv = gdc_cnv.merge(x1_1)
            gdc_cnv = gdc_cnv.sel(cols=np.isnan(gdc_cnv.data).sum(axis=0)==0)
            gdc_cnv = gdc_cnv.swap_dims({'cols': 'cnv_cols'})
            del gdc_cnv['cols']
            gdc_cnv = gdc_cnv.merge(x5_2.set_index('gdc_rows'))
            x5_3 = x5_3.rechunk((None, gdc_cnv.data.chunks[0]))
            gdc_cnv['data'] = (('gdc_rows', 'cnv_cols'),  x5_3 @ gdc_cnv.data.data.astype('float32'))
            del gdc_cnv['rows']

            x4_1 = set(crispr.depmap_rows.values)
            x4_1.intersection_update(dm_expr.depmap_rows.values)
            x4_1.intersection_update(dm_cnv.depmap_rows.values)
            x4_1 = list(x4_1)

            x4_3 = gdc_expr.expr_cols.values
            x4_3 = pd.Series(range(len(x4_3)), index=x4_3)

            x4_2 = set(dm_expr.expr_cols.values)
            x4_2.intersection_update(gdc_expr.expr_cols.values)
            x4_2 = list(x4_2)
            x4_2 = x4_3[x4_2].sort_values()
            x4_2 = list(x4_2.index)

            x4_4 = gdc_cnv.cnv_cols.values
            x4_4 = pd.Series(range(len(x4_4)), index=x4_4)

            x4_5 = set(dm_cnv.cnv_cols.values)
            x4_5.intersection_update(gdc_cnv.cnv_cols.values)
            x4_5 = list(x4_5)
            x4_5 = x4_4[x4_5].sort_values()
            x4_5 = list(x4_5.index)

            x4_6 = set(gdc_expr.gdc_rows.values)
            x4_6.intersection_update(gdc_cnv.gdc_rows.values)
            x4_6 = list(x4_6)
            x4_6 = pd.Series(range(len(gdc_expr.gdc_rows)), index=gdc_expr.gdc_rows)[x4_6].sort_values().index

            crispr = crispr.sel(depmap_rows=x4_1)
            crispr.data.data = dmlp.StandardScaler().fit_transform(crispr.data.data.astype('float32'))

            dm_cnv = dm_cnv.sel(depmap_rows=x4_1, cnv_cols=x4_5)
            dm_cnv.data.data = dmlp.StandardScaler().fit_transform(dm_cnv.data.data.astype('float32'))

            dm_expr = dm_expr.sel(depmap_rows=x4_1, expr_cols=x4_2)
            dm_expr.data.data = dmlp.StandardScaler().fit_transform(dm_expr.data.data.astype('float32'))

            gdc_expr = gdc_expr.sel(gdc_rows=x4_6, expr_cols=x4_2)
            gdc_expr.data.data = dmlp.StandardScaler().fit_transform(gdc_expr.data.data.astype('float32'))

            gdc_cnv = gdc_cnv.sel(gdc_rows=x4_6, cnv_cols=x4_5)
            gdc_cnv.data.data = dmlp.StandardScaler().fit_transform(gdc_cnv.data.data.astype('float32'))

        return x4

x4 = merge()._merge

def _perm(x):
    return x[np.random.permutation(x.shape[0]), :]

def _eval(a):
    return daa.from_array(a.compute())

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
        return SVD(x @ self.u, self.s, self.v)

    def eval(self):
        self.u = _eval(self.u)
        self.s = _eval(self.s)
        self.v = _eval(self.v)
        return self

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

        split = x4.dm_expr.depmap_rows
        split['train'] = ('depmap_rows', np.random.random(split.depmap_rows.shape)<train_split)

        self.train = [
            _eval(x4.dm_expr.sel(depmap_rows=split.train).data.data),
            _eval(x4.crispr.sel(depmap_rows=split.train).data.data),
            split.depmap_rows[split.train]
        ]

        self.test = [
            _eval(x4.dm_expr.sel(depmap_rows=~split.train).data.data),
            _eval(x4.crispr.sel(depmap_rows=~split.train).data.data),
            split.depmap_rows[~split.train]
        ]

        fit = _score3(self.train[0], self.train[1], np.s_[:dims], np.s_[:]).cut(np.s_[:]).eval()
        fit = [
            fit.mult(self.train[0]),
            fit.mult(self.test[0]),
            fit.mult(x4.gdc_expr.data.data)
        ]
        fit = [x.eval() for x in fit]
        self.fit = fit

        perm = _perm(self.train[0])
        perm_fit = _score3(perm, self.train[1], np.s_[:dims], np.s_[:]).cut(np.s_[:]).eval()
        perm_fit = perm_fit.mult(perm)
        stats = [
            ((self.train[1] - self.fit[0].usv) ** 2).mean(axis=0),
            ((self.test[1] - self.fit[1].usv) ** 2).mean(axis=0),
            ((self.train[1] - perm_fit.usv) ** 2).mean(axis=0)
        ]
        stats = [x.compute() for x in stats]
        stats = pd.DataFrame(dict(
            crispr_cols=x4.crispr.crispr_cols.values,
            train=stats[0].ravel(),
            test=stats[1].ravel(),
            rand=stats[2].ravel()
        ))
        self.stats = stats

    def data(self, idx):
        data = [
            self.fit[0].usv[:,idx], self.train[1][:,idx],
            self.fit[1].usv[:,idx], self.test[1][:,idx],
            self.fit[2].usv[:,idx]
        ]
        data = [x.compute() for x in data]
        data = [
            pd.DataFrame(dict(
                pred = data[0],
                obs = data[1],
                CCLE_Name = x4.dm_expr.CCLE_Name.loc[self.train[2]].values
            )),
            pd.DataFrame(dict(
                pred = data[2],
                obs = data[3],
                CCLE_Name = x4.dm_expr.CCLE_Name.loc[self.test[2]].values
            )),
            pd.DataFrame(dict(
                expr=data[4],
                project_id=x4.gdc_expr.project_id.values,
                is_normal=x4.gdc_expr.is_normal.values
            )),
        ]
        return data

x1 = model(0.8, 400)

x1.stats.sort_values('train').head(20)

x1.data(6665)