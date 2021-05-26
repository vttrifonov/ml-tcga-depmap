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
import xarray as xa

config.exec()

class SVD:
    def __init__(self, u, s, v):
        self.u = u
        self.s = s
        self.v = v

    @staticmethod
    def from_mat(mat, n = None, solver = 'full'):
        if n is None:
            n = min(*mat.shape)

        if solver == 'full':
            _svd = daa.linalg.svd(mat.data)
        elif solver == 'rand':
            _svd = daa.linalg.svd_compressed(mat.data, n)
        else:
            raise ValueError('unknown solver')

        _svd = (_svd[0][:,:n], _svd[1][:n], _svd[2][:n,:].T)

        svd = xa.Dataset()
        svd['u'] = ((mat.dims[0], 'pc'), _svd[0])
        svd['s'] = ('pc', _svd[1])
        svd['v'] = ((mat.dims[1], 'pc'), _svd[2])
        svd['pc'] = np.arange(n)
        svd = svd.merge(mat.coords)

        return SVD(svd.u, svd.s, svd.v)

    def cut(self, n=None):
        if n is None:
            n = np.s_[:]
        return SVD(self.u[:, n], self.s[n], self.v[:, n])

    @property
    def us(self):
        return self.u * self.s

    @property
    def vs(self):
        return self.v * self.s

    @property
    def usv(self):
        return self.us @ self.v.T

    @property
    def perm(self):
        u = _perm(self.u)
        u[u.dims[0]] = self.u[u.dims[0]]
        return SVD(u, self.s, self.v)

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
        return SVD(
            self.u.persist(),
            self.s.persist(),
            self.v.persist()
        )

    @staticmethod
    def from_xarray(x):
        return SVD(x.u, x.s, x.v)

    @property
    def lsvd(self):
        svd = SVD.from_mat(self.us)
        svd.v = self.v @ svd.v
        return svd

    @property
    def xarray(self):
        return xa.merge([self.u.rename('u'), self.s.rename('s'), self.v.rename('v')])

    @lazy_property
    def ve(self):
        ve = self.s**2
        ve = ve / ve.sum()
        return ve.rename('ve')

def _perm(x):
    return x[np.random.permutation(x.shape[0]), :]

def cache_zarr(path, data):
    if not path.exists():
        data().astype('float16').rechunk((1000, 1000)).to_zarr(str(path))
    return daa.from_zarr(zarr.open(str(path)).astype('float32'))

def cache_pickle(path, data):
    if path.exists():
        with path.open('rb') as file:
            data = pickle.load(file)
    else:
        data = data()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as file:
            pickle.dump(data, file)
    return data

def raise_(error):
    raise error

def cache_da(path, da):
    if not path.exists():
        da = da().copy()
        values = cache_zarr(path / 'values.zarr', lambda: da.data)
        args = cache_pickle(
            path/'args.pickle',
            lambda: {'name': da.name, 'dims': da.dims, 'coords': da.coords, 'attrs': da.attrs}
        )
    else:
        args = cache_pickle(path/'args.pickle', lambda: raise_(ValueError('missing args')))
        values = cache_zarr(path / 'values.zarr', lambda: raise_(ValueError('missing values')))

    da = xa.DataArray(values, **args)
    return da

def cache_ds(path, ds):
    if not path.exists():
        ds = ds().copy()
        cache_pickle(path/'non-data.pickle', lambda: ds.drop('data'))
        ds['data'] = cache_da(path/'data', lambda: ds.data)
        return ds
    else:
        ds = cache_pickle(path/'non-data.pickle', lambda: raise_(ValueError('missing non-data')))
        ds['data'] = cache_da(path/'data', lambda: raise_(ValueError('missing data')))
        return ds

def cache_svd(path, svd):
    if not path.exists():
        svd = svd()
        u = cache_da(path/'u', lambda: svd.u)
        s = cache_da(path / 's', lambda: svd.s)
        v = cache_da(path / 'v', lambda: svd.v)
    else:
        u = cache_da(path/'u', lambda: raise_(ValueError('missing u')))
        s = cache_da(path / 's', lambda: raise_(ValueError('missing s')))
        v = cache_da(path / 'v', lambda: raise_(ValueError('missing v')))
    return SVD(u, s, v)

class Mat:
    def __init__(self, mat, svd = None):
        self._mat = mat
        if svd is None:
            svd = lambda: SVD.from_mat(self.mat.data)
        self._svd = svd

    @lazy_property
    def mat(self):
        return self._mat()

    @lazy_property
    def svd(self):
        return self._svd()

    @staticmethod
    def cached(storage, mat):
        return Mat(
            lambda: cache_ds(storage, mat),
            lambda: cache_svd(
                storage / 'svd',
                lambda: SVD.from_mat(mat().data)
            )
        )

def _new_cols(ds):
    x1_1 = ds.col_entrez[['col', 'dbprimary_acc', 'display_label']]
    x1_1 = x1_1.drop_duplicates()
    x1_1 = x1_1.rename(columns={
        'col': 'cols',
        'dbprimary_acc': 'entrez',
        'display_label': 'symbol'
    })
    x1_1['new_cols'] = x1_1.symbol + ' (' + x1_1.entrez + ')'
    x1_1['n'] = x1_1.groupby('new_cols').new_cols.transform('size')
    x1_1 = x1_1.query('n==1 | cols.str.find("ENSGR")<0').copy()
    x1_1['n'] = x1_1.groupby('new_cols').new_cols.transform('size')
    x1_1 = x1_1.query('n==1').copy()
    del x1_1['n']
    x1_1 = x1_1.set_index('cols').to_xarray()
    return x1_1

def _new_rows(x5_1):
    x5_1 = x5_1.to_dataframe().reset_index()
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
    return [x5_2, x5_3]

class merge:
    @lazy_property
    def storage(self):
        return config.cache / 'merge'

    @lazy_property
    def crispr1(self):
        crispr = depmap_crispr.data.copy()
        crispr = crispr.merge(depmap_crispr.row_annot.set_index('rows'), join='inner')
        crispr = crispr.merge(depmap_crispr.col_map_location.set_index('cols'), join='inner')
        crispr = crispr.sel(cols=np.isnan(crispr.mat).sum(axis=0)==0)
        crispr['mat'] = (('rows', 'cols'), crispr.mat.data.rechunk(-1, 1000))
        crispr = crispr.rename({'mat': 'data'})
        return crispr

    @lazy_property
    def dm_expr1(self):
        dm_expr = depmap_expr.data.copy()
        dm_expr = dm_expr.merge(depmap_expr.row_annot.set_index('rows'), join='inner')
        dm_expr = dm_expr.merge(depmap_expr.col_map_location.set_index('cols'), join='inner')
        dm_expr = dm_expr.sel(cols=np.isnan(dm_expr.mat).sum(axis=0)==0)
        dm_expr['mat'] = (('rows', 'cols'), dm_expr.mat.data.rechunk(-1, 1000))
        dm_expr['mean'] = dm_expr.mat.mean(axis=0)
        dm_expr = dm_expr.sel(cols=dm_expr['mean']>1.5)
        dm_expr = dm_expr.rename({'mat': 'data'})
        return dm_expr

    @lazy_property
    def dm_cnv1(self):
        dm_cnv = depmap_cnv.data.copy()
        dm_cnv = dm_cnv.merge(depmap_cnv.row_annot.set_index('rows'), join='inner')
        dm_cnv = dm_cnv.merge(depmap_cnv.col_map_location.set_index('cols'), join='inner')
        dm_cnv = dm_cnv.sel(cols=np.isnan(dm_cnv.mat).sum(axis=0)==0)
        dm_cnv['mat'] = (('rows', 'cols'), dm_cnv.mat.data.rechunk(-1, 1000))
        dm_cnv = dm_cnv.rename({'mat': 'data'})
        return dm_cnv

    @lazy_property
    def gdc_expr1(self):
        new_cols = _new_cols(_gdc_expr)
        new_rows = _new_rows(_gdc_expr.xarray[['rows', 'project_id', 'is_normal', 'case_id', 'sample_id']])
        gdc_expr = _gdc_expr.xarray[['data', 'rows', 'cols']]
        gdc_expr = gdc_expr.sel(cols=new_cols.cols)
        gdc_expr = gdc_expr.merge(new_cols)
        gdc_expr = gdc_expr.swap_dims({'cols': 'new_cols'}).drop('cols').rename({'new_cols': 'cols'})
        gdc_expr = gdc_expr.merge(new_rows[0].set_index('new_rows'))
        new_rows[1] = new_rows[1].rechunk((None, gdc_expr.data.chunks[0]))
        gdc_expr['data'] = (('new_rows', 'cols'),  new_rows[1] @ gdc_expr.data.data.astype('float32'))
        gdc_expr = gdc_expr.drop('rows').rename({'new_rows': 'rows'})
        gdc_expr['mean'] = gdc_expr.data.mean(axis=0).compute()
        gdc_expr = gdc_expr.sel(cols=gdc_expr['mean']>(-7))
        return gdc_expr

    @lazy_property
    def gdc_cnv1(self):
        new_cols = _new_cols(_gdc_cnv)
        new_rows = _new_rows(_gdc_cnv.xarray[['rows', 'project_id','case_id', 'sample_id']])
        gdc_cnv = _gdc_cnv.xarray[['data', 'rows', 'cols']]
        gdc_cnv = gdc_cnv.sel(cols=new_cols.cols)
        gdc_cnv = gdc_cnv.merge(new_cols)
        gdc_cnv = gdc_cnv.sel(cols=np.isnan(gdc_cnv.data).sum(axis=0)==0)
        gdc_cnv = gdc_cnv.swap_dims({'cols': 'new_cols'}).drop('cols').rename({'new_cols': 'cols'})
        gdc_cnv = gdc_cnv.merge(new_rows[0].set_index('new_rows'))
        new_rows[1] = new_rows[1].rechunk((None, gdc_cnv.data.chunks[0]))
        gdc_cnv['data'] = (('new_rows', 'cols'),  new_rows[1] @ gdc_cnv.data.data.astype('float32'))
        gdc_cnv = gdc_cnv.drop('rows').rename({'new_rows': 'rows'})
        return gdc_cnv

    @lazy_property
    def _merge(self):
        crispr = self.crispr1
        dm_expr = self.dm_expr1
        dm_cnv = self.dm_cnv1

        gdc_expr = self.gdc_expr1
        gdc_expr = gdc_expr.sel(cols=gdc_expr.cols.isin(dm_expr.cols))

        gdc_cnv = self.gdc_cnv1
        gdc_cnv = gdc_cnv.sel(cols=gdc_cnv.cols.isin(dm_cnv.cols))

        depmap_rows = set(crispr.rows.values)
        depmap_rows.intersection_update(dm_expr.rows.values)
        depmap_rows.intersection_update(dm_cnv.rows.values)
        depmap_rows = list(depmap_rows)

        expr_cols = set(dm_expr.cols.values)
        expr_cols.intersection_update(gdc_expr.cols.values)
        expr_cols = list(expr_cols)
        expr_cols = pd.Series(range(len(gdc_expr.cols)), index=gdc_expr.cols)[expr_cols].sort_values()
        expr_cols = list(expr_cols.index)

        cnv_cols = set(dm_cnv.cols.values)
        cnv_cols.intersection_update(gdc_cnv.cols.values)
        cnv_cols = list(cnv_cols)
        cnv_cols = pd.Series(range(len(gdc_cnv.cols)), index=gdc_cnv.cols)[cnv_cols].sort_values()
        cnv_cols = list(cnv_cols.index)

        gdc_rows = set(gdc_expr.rows.values)
        gdc_rows.intersection_update(gdc_cnv.rows.values)
        gdc_rows = list(gdc_rows)
        gdc_rows = pd.Series(range(len(gdc_expr.rows)), index=gdc_expr.rows)[gdc_rows].sort_values().index

        data = dict(
            crispr=crispr.sel(rows=depmap_rows),
            dm_cnv=dm_cnv.sel(rows=depmap_rows, cols=cnv_cols),
            dm_expr=dm_expr.sel(rows=depmap_rows, cols=expr_cols),
            gdc_cnv=gdc_cnv.sel(rows=gdc_rows, cols=cnv_cols),
            gdc_expr=gdc_expr.sel(rows=gdc_rows, cols=expr_cols)
        )

        for v in data.values():
            v.data.data = dmlp.StandardScaler().fit_transform(v.data.data.astype('float32'))

        return SimpleNamespace(**data)

    @lazy_property
    def crispr(self):
        return Mat.cached(self.storage/'crispr', lambda: self._merge.crispr)

    @lazy_property
    def dm_cnv(self):
        return Mat.cached(self.storage/'dm_cnv', lambda: self._merge.dm_cnv)

    @lazy_property
    def dm_expr(self):
        return Mat.cached(self.storage/'dm_expr', lambda: self._merge.dm_expr)

    @lazy_property
    def gdc_cnv(self):
        return Mat.cached(self.storage/'gdc_cnv', lambda: self._merge.gdc_cnv)

    @lazy_property
    def gdc_expr(self):
        return Mat.cached(self.storage/'gdc_expr', lambda: self._merge.gdc_expr)

def concat(x):
    x = (y.data.copy() for y in x)
    x = (y.assign_coords({'cols': str(i) + ':' + y.cols}) for i, y in enumerate(x))
    x = xa.concat(x, 'cols')
    x = xa.Dataset().assign(data=x)
    return x

def concat1(x):
    x = [y.copy() for y in x]
    x = [y.assign_coords({'cols': str(i) + ':' + y.cols.astype('str').to_series()}) for i, y in enumerate(x)]
    x = xa.concat(x, 'cols')
    return x

class model:
    def __init__(self, x, y, z, reg):
        self.reg = reg
        self.x = x
        self.y = y
        self.z = z

        fit = x.train.cut(reg[1]).inv(reg[0])
        fit = fit.rmult(y.train.data)
        fit = fit.persist()
        fit = dict(
            train=fit.lmult(x.train.usv),
            test=fit.lmult(x.test),
            pred=fit.lmult(z.data)
        )
        fit = {k: v.persist() for k, v in fit.items()}
        self.fit = SimpleNamespace(**fit)

        perm = x.train.perm
        perm_fit = perm.cut(reg[1]).inv(reg[0])
        perm_fit = perm_fit.rmult(y.train.data)
        perm_fit = perm_fit.lmult(perm.usv)
        perm_fit = perm_fit.persist()

        stats = dict(
            train=((y.train.data - self.fit.train.usv) ** 2).mean(axis=0),
            test=((y.test.data - self.fit.test.usv) ** 2).mean(axis=0),
            rand=((y.train.data - perm_fit.usv) ** 2).mean(axis=0)
        )
        stats = {k: v.compute().data.ravel() for k, v in stats.items()}
        stats = pd.DataFrame(dict(
            cols=y.train.cols.values,
            **stats
        ))
        self.stats = stats

    def data(self, idx):
        data = [
            self.fit.train.usv[:,idx], self.y.train.data[:,idx],
            self.fit.test.usv[:,idx], self.y.test.data[:,idx],
            self.fit.pred.usv[:,idx]
        ]
        data = [x.compute() for x in data]
        data = [
            pd.DataFrame(dict(
                pred = data[0],
                obs = data[1],
                CCLE_Name = self.y.train.CCLE_Name.values
            )),
            pd.DataFrame(dict(
                pred = data[2],
                obs = data[3],
                CCLE_Name = self.y.test.CCLE_Name.values
            )),
            pd.DataFrame(dict(
                expr=data[4],
                project_id=self.z.project_id.values,
                is_normal=self.z.is_normal.values
            )),
        ]
        return data

    @staticmethod
    def splitx(x, split):
        return SimpleNamespace(
            x = x,
            train = SVD.from_xarray(x.svd.xarray.sel(rows=split.train).rename({'pc': '__tmp_pc__'})).lsvd.persist(),
            test = x.mat.sel(rows=~split.train).data.persist()
        )

    @staticmethod
    def splity(x, split):
        return SimpleNamespace(
            x = x,
            train = x.mat.sel(rows=split.train).persist(),
            test = x.mat.sel(rows=~split.train).persist()
        )

