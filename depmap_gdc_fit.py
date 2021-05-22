import pandas as pd
import numpy as np
import dask_ml.preprocessing as dmlp
from types import SimpleNamespace
import dask.array as daa
from depmap_crispr import crispr as depmap_crispr
from depmap_expr import expr as depmap_expr
from depmap_cnv import cnv as depmap_cnv
from gdc_expr import expr as _gdc_expr
from gdc_cnv import cnv as _gdc_cnv
import sparse as sp
from common.defs import lazy_property
from helpers import config
import pickle
import zarr

config.exec()

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

    def inv(self, l = 0):
        return SVD(self.v, self.s/(l + self.s**2), self.u)

    @property
    def T(self):
        return SVD(self.v, self.s, self.u)

    def lmult(self, x):
        return SVD(x @ self.u, self.s, self.v)

    def rmult(self, x):
        return SVD(self.u, self.s, x.T @ self.v)

    def persist(self):
        self.u = self.u.persist()
        self.s = self.s.persist()
        self.v = self.v.persist()
        return self

    @staticmethod
    def from_xarray(x):
        return SVD(x.u.data, x.s.data, x.v.data)

    @property
    def svd(self):
        svd = SVD.from_data(self.us)
        svd.v = self.v @ svd.v
        return svd

    def mult_svd(self, x):
        svd = SVD.from_data(self.vs.T @ x.us)
        svd.u = self.u @ svd.u
        svd.v = x.v @ svd.v
        return svd

def _cache(path, x):
    meta = path/'meta.pickle'
    data = str(path/'data.zarr')

    if not path.exists():
        x = x()
        x.data.data.astype('float16').rechunk((1000, 1000)).to_zarr(data)
        with meta.open('wb') as file:
            pickle.dump((x.data.dims, x.drop('data')), file)

    with meta.open('rb') as file:
        p = pickle.load(file)
    x = p[1]
    x['data'] = (p[0], daa.from_zarr(zarr.open(data).astype('float32')))
    return x

def _cache_svd(path, x):
    meta = path/'meta.pickle'
    u = str(path/'u.zarr')
    v = str(path/'v.zarr')

    if not path.exists():
        x = x()
        svd = SVD.from_data(x.data.data.astype('float32'))
        svd.u.astype('float16').rechunk((1000, 1000)).to_zarr(u)
        svd.v.astype('float16').rechunk((1000, 1000)).to_zarr(v)
        x['s'] = ('pc', svd.s.persist())
        with meta.open('wb') as file:
            pickle.dump((x.data.dims, x.drop('data')), file)

    with meta.open('rb') as file:
        p = pickle.load(file)
    x = p[1]
    x['u'] = ((p[0][0], 'pc'), daa.from_zarr(zarr.open(u).astype('float32')))
    x['v'] = ((p[0][1], 'pc'), daa.from_zarr(zarr.open(v).astype('float32')))
    x['pc'] = np.arange(len(x.s))
    return x

def _perm(x):
    return x[np.random.permutation(x.shape[0]), :]

class merge:
    @lazy_property
    def storage(self):
        return config.cache / 'merge'

    @lazy_property
    def _merge(self):
        crispr = depmap_crispr.data.copy()
        crispr = crispr.sel(cols=np.isnan(crispr.mat).sum(axis=0)==0)
        crispr['mat'] = (('rows', 'cols'), crispr.mat.data.rechunk(-1, 1000))
        crispr = crispr.rename({'mat': 'data'})

        dm_expr = depmap_expr.data.copy()
        dm_expr = dm_expr.merge(
            depmap_expr.release.samples.rename(columns={'DepMap_ID': 'rows'}).set_index('rows').to_xarray(),
            join='inner'
        )
        dm_expr = dm_expr.sel(cols=np.isnan(dm_expr.mat).sum(axis=0)==0)
        dm_expr['mat'] = (('rows', 'cols'), dm_expr.mat.data.rechunk(-1, 1000))
        dm_expr['mean'] = dm_expr.mat.mean(axis=0)
        dm_expr = dm_expr.sel(cols=dm_expr['mean']>1.5)
        dm_expr = dm_expr.rename({'mat': 'data'})

        dm_cnv = depmap_cnv.data.copy()
        dm_cnv = dm_cnv.sel(cols=np.isnan(dm_cnv.mat).sum(axis=0)==0)
        dm_cnv['mat'] = (('rows', 'cols'), dm_cnv.mat.data.rechunk(-1, 1000))
        dm_cnv = dm_cnv.rename({'mat': 'data'})

        x1_1 = _gdc_expr.col_entrez[['col', 'dbprimary_acc', 'display_label']]
        x1_1 = x1_1.drop_duplicates()
        x1_1 = x1_1.rename(columns={
            'col': 'cols',
            'dbprimary_acc': 'entrez',
            'display_label': 'symbol'
        })
        x1_1['new_cols'] = x1_1.symbol + ' (' + x1_1.entrez + ')'
        x1_1 = x1_1.query('new_cols.isin(@dm_expr.cols.values)').copy()
        x1_1['n'] = x1_1.groupby('new_cols').new_cols.transform('size')
        x1_1 = x1_1.query('n==1 | cols.str.find("ENSGR")<0').copy()
        x1_1['n'] = x1_1.groupby('new_cols').new_cols.transform('size')
        x1_1 = x1_1.query('n==1').copy()
        del x1_1['n']
        x1_1 = x1_1.set_index('cols').to_xarray()

        x5_1 = _gdc_expr.xarray[['rows', 'project_id', 'is_normal', 'case_id', 'sample_id']].to_dataframe().reset_index()
        x5_1 = x5_1.rename(columns={'sample_id': 'new_rows'})
        x5_2 = x5_1.drop(columns=['rows']).drop_duplicates()
        x5_3 = x5_1[['rows', 'new_rows']].copy()
        x5_3['w'] = x5_3.groupby('new_rows').new_rows.transform('size')
        x5_3['w'] = 1/x5_3.w
        x5_3['rows'] = pd.Series(range(x5_1.shape[0]), index=x5_1.rows)[x5_3.rows.to_numpy()].to_numpy()
        x5_3['new_rows'] = pd.Series(range(x5_2.shape[0]), index=x5_2.new_rows)[x5_3.new_rows.to_numpy()].to_numpy()
        x5_3 = daa.from_array(
            sp.COO(x5_3[['new_rows', 'rows']].to_numpy().T, x5_3.w.astype('float32')),
            chunks=(1000,-1)
        )

        gdc_expr = _gdc_expr.xarray[['data', 'rows', 'cols']]
        gdc_expr = gdc_expr.sel(cols=x1_1.cols)
        gdc_expr = gdc_expr.merge(x1_1)
        gdc_expr = gdc_expr.swap_dims({'cols': 'new_cols'}).drop('cols').rename({'new_cols': 'cols'})
        gdc_expr = gdc_expr.merge(x5_2.set_index('new_rows'))
        x5_3 = x5_3.rechunk((None, gdc_expr.data.chunks[0]))
        gdc_expr['data'] = (('new_rows', 'cols'),  x5_3 @ gdc_expr.data.data.astype('float32'))
        gdc_expr = gdc_expr.drop('rows').rename({'new_rows': 'rows'})
        gdc_expr['mean'] = gdc_expr.data.mean(axis=0).compute()
        gdc_expr = gdc_expr.sel(cols=gdc_expr['mean']>(-7))

        x1_1 = _gdc_cnv.col_entrez[['col', 'dbprimary_acc', 'display_label']].copy()
        x1_1 = x1_1.drop_duplicates()
        x1_1 = x1_1.rename(columns={
            'col': 'cols',
            'dbprimary_acc': 'entrez',
            'display_label': 'symbol'
        })
        x1_1['new_cols'] = x1_1.symbol + ' (' + x1_1.entrez + ')'
        x1_1 = x1_1.query('new_cols.isin(@dm_cnv.cols.values)').copy()
        x1_1['n'] = x1_1.groupby('new_cols').new_cols.transform('size')
        x1_1 = x1_1.query('n==1 | cols.str.find("ENSGR")<0').copy()
        x1_1['n'] = x1_1.groupby('new_cols').new_cols.transform('size')
        x1_1 = x1_1.query('n==1').copy()
        del x1_1['n']
        x1_1 = x1_1.set_index('cols').to_xarray()

        x5_1 = _gdc_cnv.xarray[['rows', 'project_id', 'case_id', 'sample_id']].to_dataframe().reset_index()
        x5_1 = x5_1.rename(columns={'sample_id': 'new_rows'})
        x5_2 = x5_1.drop(columns=['rows']).drop_duplicates()
        x5_3 = x5_1[['rows', 'new_rows']].copy()
        x5_3['w'] = x5_3.groupby('new_rows').new_rows.transform('size')
        x5_3['w'] = 1/x5_3.w
        x5_3['rows'] = pd.Series(range(x5_1.shape[0]), index=x5_1.rows)[x5_3.rows.to_numpy()].to_numpy()
        x5_3['new_rows'] = pd.Series(range(x5_2.shape[0]), index=x5_2.new_rows)[x5_3.new_rows.to_numpy()].to_numpy()
        x5_3 = daa.from_array(
            sp.COO(x5_3[['new_rows', 'rows']].to_numpy().T, x5_3.w.astype('float32')),
            chunks=(1000,-1)
        )

        gdc_cnv = _gdc_cnv.xarray[['data', 'rows', 'cols']]
        gdc_cnv = gdc_cnv.sel(cols=x1_1.cols)
        gdc_cnv = gdc_cnv.merge(x1_1)
        gdc_cnv = gdc_cnv.sel(cols=np.isnan(gdc_cnv.data).sum(axis=0)==0)
        gdc_cnv = gdc_cnv.swap_dims({'cols': 'new_cols'}).drop('cols').rename({'new_cols': 'cols'})
        gdc_cnv = gdc_cnv.merge(x5_2.set_index('new_rows'))
        x5_3 = x5_3.rechunk((None, gdc_cnv.data.chunks[0]))
        gdc_cnv['data'] = (('new_rows', 'cols'),  x5_3 @ gdc_cnv.data.data.astype('float32'))
        gdc_cnv = gdc_cnv.drop('rows').rename({'new_rows': 'rows'})

        x4_1 = set(crispr.rows.values)
        x4_1.intersection_update(dm_expr.rows.values)
        x4_1.intersection_update(dm_cnv.rows.values)
        x4_1 = list(x4_1)

        x4_3 = gdc_expr.cols.values
        x4_3 = pd.Series(range(len(x4_3)), index=x4_3)

        x4_2 = set(dm_expr.cols.values)
        x4_2.intersection_update(gdc_expr.cols.values)
        x4_2 = list(x4_2)
        x4_2 = x4_3[x4_2].sort_values()
        x4_2 = list(x4_2.index)

        x4_4 = gdc_cnv.cols.values
        x4_4 = pd.Series(range(len(x4_4)), index=x4_4)

        x4_5 = set(dm_cnv.cols.values)
        x4_5.intersection_update(gdc_cnv.cols.values)
        x4_5 = list(x4_5)
        x4_5 = x4_4[x4_5].sort_values()
        x4_5 = list(x4_5.index)

        x4_6 = set(gdc_expr.rows.values)
        x4_6.intersection_update(gdc_cnv.rows.values)
        x4_6 = list(x4_6)
        x4_6 = pd.Series(range(len(gdc_expr.rows)), index=gdc_expr.rows)[x4_6].sort_values().index

        crispr = crispr.sel(rows=x4_1)
        crispr.data.data = dmlp.StandardScaler().fit_transform(crispr.data.data.astype('float32'))

        dm_cnv = dm_cnv.sel(rows=x4_1, cols=x4_5)
        dm_cnv.data.data = dmlp.StandardScaler().fit_transform(dm_cnv.data.data.astype('float32'))

        dm_expr = dm_expr.sel(rows=x4_1, cols=x4_2)
        dm_expr.data.data = dmlp.StandardScaler().fit_transform(dm_expr.data.data.astype('float32'))

        gdc_expr = gdc_expr.sel(rows=x4_6, cols=x4_2)
        gdc_expr.data.data = dmlp.StandardScaler().fit_transform(gdc_expr.data.data.astype('float32'))

        gdc_cnv = gdc_cnv.sel(rows=x4_6, cols=x4_5)
        gdc_cnv.data.data = dmlp.StandardScaler().fit_transform(gdc_cnv.data.data.astype('float32'))

        return SimpleNamespace(
            crispr = crispr,
            dm_cnv = dm_cnv,
            dm_expr = dm_expr,
            gdc_expr = gdc_expr,
            gdc_cnv = gdc_cnv
        )

    @lazy_property
    def crispr(self):
        return _cache_svd(self.storage/'crispr'/'svd', lambda: self._merge.crispr)

    @lazy_property
    def dm_cnv(self):
        return _cache_svd(self.storage/'dm_cnv'/'svd', lambda: self._merge.dm_cnv)

    @lazy_property
    def dm_expr(self):
        return _cache_svd(self.storage/'dm_expr'/'svd', lambda: self._merge.dm_expr)

    @lazy_property
    def gdc_expr(self):
        return _cache(self.storage/'gdc_expr', lambda: self._merge.gdc_expr)

    @lazy_property
    def gdc_cnv(self):
        return _cache(self.storage/'gdc_cnv', lambda: self._merge.gdc_cnv)

class model:
    def __init__(self, x, y, z, reg):
        self.reg = reg
        self.x = x
        self.y = y
        self.z = z

        fit = x.train.cut(reg[1]).inv(reg[0])
        fit = fit.rmult(y.train)
        fit = fit.persist()
        fit = [
            fit.lmult(x.train.usv),
            fit.lmult(x.test),
            fit.lmult(z.data.data)
        ]
        fit = [x.persist() for x in fit]
        self.fit = fit

        perm = x.train.perm
        perm_fit = perm.cut(reg[1]).inv(reg[0])
        perm_fit = perm_fit.rmult(y.train)
        perm_fit = perm_fit.lmult(perm.usv)
        perm_fit = perm_fit.persist()

        stats = [
            ((y.train - self.fit[0].usv) ** 2).mean(axis=0),
            ((y.test - self.fit[1].usv) ** 2).mean(axis=0),
            ((y.train - perm_fit.usv) ** 2).mean(axis=0)
        ]
        stats = [x.compute() for x in stats]
        stats = pd.DataFrame(dict(
            cols=y.x.cols.values,
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
                CCLE_Name = self.x.CCLE_Name.loc[self.merge.split.train].values
            )),
            pd.DataFrame(dict(
                pred = data[2],
                obs = data[3],
                CCLE_Name = self.x.CCLE_Name.loc[~self.merge.split.train].values
            )),
            pd.DataFrame(dict(
                expr=data[4],
                project_id=self.z.project_id.values,
                is_normal=self.z.is_normal.values
            )),
        ]
        return data

