# %%
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9
from common.dir import cached_property, Dir
from common.defs import lazy_property
from types import SimpleNamespace as namespace
import types
import dask.array as daa
import zarr
from pathlib import Path
import xarray as xa
import pickle
import numpy as np
import pandas as pd
import importlib
import dask_ml.preprocessing as dmlp
import plotly.express as px
import dask
from helpers import config
from svd import SVD
from merge import merge
from common.caching import compose, lazy, XArrayCache, PickleCache

from common.caching import FileCache

class DictCache(FileCache):
    def __init__(self, *default, **elem):
        super().__init__(True, '')
        self.default = default[0] if len(default)>0 else None
        self.elem = elem

    def store(self, data, storage):
        for k, v in data.items():
            s = self.elem.get(k, self.default)
            if s is not None:
                s.store(v, storage/str(k))

    def restore(self, storage):
        data = {}
        for k in storage.glob('*'):
            k = k.name
            s = self.elem.get(k, self.default)
            if s is not None:
                data[k] = s.restore(storage/k)
        return data

class ArrayCache(FileCache):
    def __init__(self, elem):
        super().__init__(True, ext='')
        self.elem = elem

    def store(self, data, storage):
        for i, v in enumerate(data):
            self.elem.store(v, storage/str(i))

    def restore(self, storage):
        i = [int(x.name) for x in storage.glob('*')]
        return [
            self.elem.restore(storage/str(i))
            for i in sorted(i)
        ]
    
class ClassCache(DictCache):
    def __init__(self, cls, **elem):
        super().__init__(**elem)
        self.cls = cls

    def store(self, data, storage):
        super().store(data.__dict__, storage)

    def restore(self, storage):
        return self.cls(**super().restore(storage))

# %%

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

config.exec()

storage = config.cache/'playground11'

# %%

def _smooth(x, window_len=11, window='hanning'):
    if len(x) < window_len:
        return x

    if window_len<3:
        return x

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    y = y[window_len-1:-(window_len-1)]
    return y

def _fft(x):
    fft = daa.hstack([x, daa.fliplr(x[:, 1:])]).rechunk((None, -1))
    fft = daa.fft.rfft(fft)
    fft = daa.real(fft)
    return fft

def _rfft(x, n):
    rfft = x[:, :n]
    if x.shape[1] > n:
        rfft = daa.hstack([rfft, daa.zeros((x.shape[0], x.shape[1] - n))])
    rfft = rfft.rechunk((None, -1))
    rfft = daa.fft.irfft(rfft)
    rfft = rfft[:, :x.shape[1]]
    return rfft

def _fft1(d1, cutoff, g, cols):
    f = xa.Dataset()

    d = d1.groupby(g)
    d = [
        xa.apply_ufunc(
            _fft, x,
            input_core_dims = [[cols]], output_core_dims=[['freq']],
            dask='allowed'
        ).assign_coords({
            'fft_'+g: ('freq', x[g]),
            'fft_freq': ('freq', np.arange(x.shape[1]))
        }).assign_coords({
            'freq': lambda x: ('freq', x['fft_'+g] + ':' + x.fft_freq.astype(str))
        })
        for _, x in d
    ]
    d = xa.concat(d, 'freq')
    d.data = d.data.rechunk(d1.data.chunks)
    f['fft'] = d
    print('hi1')

    d = f.fft.groupby('fft_' + g)
    d = [
        xa.apply_ufunc(
            _rfft, x, min(cutoff, x.shape[1]),
            input_core_dims = [['freq'], []], output_core_dims=[[cols]],
            dask='allowed'
        ).assign_coords({
            cols: d1[cols].sel({cols: d1[g]==l})
        })
        for l, x in d
    ]
    d = xa.concat(d, cols)
    d = d.sel({cols: d1.cols})
    d.data = d.data.rechunk(d1.data.chunks)
    f['rfft'] = d
    f['fft_resid'] = d1 - f.rfft
    print('hi2')

    return f

def _fft2(x1, g):
    x2 = x1[g].to_numpy()
    x2 = np.hstack([True, x2[1:]!=x2[:-1]])
    x3 = x1.index.to_numpy()
    x3 = np.hstack([True, x3[1:]!=(x3[:-1]+1)])
    return  np.cumsum(x2 | x3)
    #x1['f'] = np.cumsum(x2 | x3)
    #x4 = np.zeros(d1[cols].shape[0])
    #x4[x1.index] = x1.f
    #f['fft_group'] = (cols, x4.astype(int))
    #print('hi3')

def _svd1(d, cutoff, cols, rand = False):
    x3 = d.fft_resid
    x3 = x3.assign_coords(fft_group=d.fft_group)
    x3 = x3.sel({cols: ~x3.fft_group.isnull()})
    x4 = d.fft.sel(freq=d.fft.fft_freq<cutoff)
    x4 = x4.rename({'freq': 'cols'}).drop(['fft_arm', 'fft_freq'])
    x4['fft_group'] = ('cols', np.zeros(x4.shape[1]))
    x3 = xa.concat([x3, x4], 'cols')
    x3 = dmlp.StandardScaler().fit_transform(x3)
    x3['fft_group'] = x3.fft_group.astype('int')
    if rand:
        x3.data = daa.apply_along_axis(np.random.permutation, 0, x3.data, shape=(x3.shape[0],), dtype=x3.dtype)
    x3 = x3.groupby('fft_group')
    x3 = {k: SVD.from_mat(x) for k, x in x3}
    return x3

def _fit(x1, x3, cutoff):
    x3 = SVD.from_mat(x3).persist()
    x2 = x3.cut(np.s_[:cutoff]).u
    x2 = ((x1 - x2 @ (x2.T @ x1)) ** 2).mean(axis=0, keepdims=True)
    x2 = daa.log10(1 - x2 + 1e-7)
    x4 = x3.perm.cut(np.s_[:cutoff]).u
    x4 = ((x1 - x4 @ (x4.T @ x1)) ** 2).mean(axis=0, keepdims=True)
    x4 = daa.log10(1 - x4 + 1e-7)
    x4 = dmlp.StandardScaler().fit(x4.T)
    x2 = x4.transform(x2.T).T.squeeze().persist()
    f = xa.Dataset()
    f['rand_mean'] = x4.mean_[0]
    f['rand_var'] = x4.var_[0]
    f['r2'] = x2
    return f

def _smooth1(d1, frac, g, cols):
    #d = m.dm_cnv.data.assign_coords(arm=m.dm_cnv.arm)
    d = d1.groupby(g)
    d = d.map(lambda x: (
        xa.apply_ufunc(
            _smooth, x, int(frac*x.shape[1]),
            input_core_dims = [[cols], []], output_core_dims=[[cols]],
            dask='parallelized', vectorize=True
        )
    ))
    d = d.drop(g)
    d.data = d.data.rechunk((None, -1))

    f = xa.Dataset()
    f['smooth'] = d
    f['smooth_resid'] = d1 - d
    return f

def _scale(d):
    fit = dmlp.StandardScaler().fit(d)
    d['mean'] = (d.dims[1], fit.mean_)
    d['var'] = (d.dims[1], fit.var_)
    d = fit.transform(d)
    return d

def _scale1(d, rows=['rows']):
    cols = list(set(d.dims)-set(rows))
    d['center'] = (cols, d.mean(dim=rows))
    d = d - d.center
    d['scale'] = (cols, np.sqrt((d**2).mean(dim=rows)))
    d = d/d.scale
    return d

def plot_fft_resid(x1):
    px.scatter(
        (x1.fft_resid ** 2).mean(axis=0).rename('mean'). \
            assign_coords(arm=x1.arm, cyto=x1.cyto). \
            to_dataframe().reset_index().reset_index(). \
            query('mean>=0')
        ,
        x='index', y='mean', color='arm',
        hover_data=['cyto', 'cols']
    ).show()

def plot_fft(x1):
    px.scatter(
        x1.rfft[0, :]. \
            assign_coords(
            data=x1.data[0, :],
            arm=x1.arm, cyto=x1.cyto
        ). \
            to_dataframe().reset_index().reset_index(),
        x='index', y=['rfft', 'data'], color='arm',
        hover_data=['cyto', 'cols']
    ).show()

# %%

class _playground11:
    pass

def _():
    def __init__(self, name, train_split_ratio):
        self._train_split_ratio = train_split_ratio
        self.name = name
    _playground11.__init__ = __init__

    @lazy_property
    def storage(self):
        return Dir(config.cache / 'playground11' / self.name)
    _playground11.storage = storage
    #del self.__lazy__storage

    @lazy_property
    @cached_property(type=Dir.pickle)
    def train_split_ratio(self):
        return self._train_split_ratio
    _playground11.train_split_ratio = train_split_ratio
    #del self.__lazy__train_split_ratio

    @lazy_property
    @cached_property(type=Dir.pickle)
    def train_split(self):
        rows = merge.crispr.rows
        rows['train'] = ('rows', np.random.random(rows.shape[0])<=self.train_split_ratio)
        return rows
    _playground11.train_split = train_split
    #del self.__lazy__train_split

    @lazy_property
    def crispr(self):
        d = merge.crispr.copy()
        d = d.sel(rows=self.train_split.train)
        d['data'] = _scale(d.data.astype('float32'))
        d = d.reset_coords(['mean', 'var', 'train'])
        return d
    _playground11.crispr = crispr
    #del self.__lazy__crispr

    @lazy_property
    def dm_expr(self):
        d = merge.dm_expr.copy()
        d = d.sel(rows=self.train_split.train)
        d['data'] = _scale(d.data.astype('float32'))
        d = d.reset_coords(['mean', 'var', 'train'])
        return d
    _playground11.dm_expr = dm_expr
    #del self.__lazy__dm_expr

    @lazy_property
    def dm_cnv(self):
        d = merge.dm_cnv.copy()
        d = d.sel(rows=self.train_split.train)
        d['txMid'] = (d.txStart+d.txEnd)/2
        d['data'] = _scale(d.data.astype('float32'))
        d = d.sortby(['chrom', 'txMid'])
        d = d.reset_coords(['mean', 'var', 'train'])
        return d
    _playground11.dm_cnv = dm_cnv
    #del self.__lazy__dm_cnv

    @lazy_property
    def gdc_cnv(self):
        d = merge.gdc_cnv.copy()
        d['txMid'] = (d.txStart + d.txEnd) / 2
        d.data.data = d.data.data.rechunk((None, -1))
        d['data'] = _scale(d.data.astype('float32'))
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            d = d.sortby(['chrom', 'txMid'])
        d = d.reset_coords(['mean', 'var'])
        return d
    _playground11.gdc_cnv = gdc_cnv
    #del self.__lazy__gdc_cnv

    @lazy_property
    def dm_cnv_fft(self):
        d = self.dm_cnv.data.assign_coords(arm=self.dm_cnv.arm)
        d = _fft1(d, 10, 'arm', 'cols')
        d = d.drop('arm')
        d = d.persist()
        return d
    _playground11.dm_cnv_fft = dm_cnv_fft
    #del self.__lazy__dm_cnv_fft

    @lazy_property
    def dm_cnv_fft_stats(self):
        d = self.dm_cnv_fft

        d1 = xa.Dataset()
        d1['fft_mean'] = d.fft.mean(axis=0)
        d1['fft_var'] = d.fft.var(axis=0)

        d2 = xa.Dataset()
        d2['fft_resid_mean'] = d.fft_resid.mean(axis=0)
        d2['fft_resid_var'] = d.fft_resid.var(axis=0)

        d = xa.merge([d1, d2])
        return d
    _playground11.dm_cnv_fft_stats = dm_cnv_fft_stats
    #del self.__lazy__dm_cnv_fft_stats

    @lazy_property
    def gdc_cnv_fft(self):
        d = self.gdc_cnv.data.assign_coords(arm=self.gdc_cnv.arm)
        d = _fft1(d, 10, 'arm', 'cols')
        d = d.drop('arm')
        d = d.persist()
        return d
    _playground11.gdc_cnv_fft = gdc_cnv_fft
    #del self.__lazy__gdc_cnv_fft

    @lazy_property
    @cached_property(type=Dir.pickle)
    def dm_cnv_fft_group(self):
        x1 = (self.dm_cnv_fft.fft_resid**2).mean(axis=0)
        x2 = (self.gdc_cnv_fft.fft_resid**2).mean(axis=0)
        x3 = xa.merge([x1.rename('dm'), x2.rename('gdc')])
        x3['arm'] = self.dm_cnv.arm
        x3 = x3.to_dataframe().reset_index()

        x1 = self.dm_cnv_fft.fft_resid.data
        x1 = x1/daa.sqrt((x1**2).sum(axis=0, keepdims=True))
        x1 = x1[:,1:] * x1[:,:-1]
        x1 = x1.sum(axis=0)
        x1 = x1.compute()
        x3['dm_cor'] = np.hstack([x1, [0]])

        x1 = self.gdc_cnv_fft.fft_resid.data
        x1 = x1/daa.sqrt((x1**2).sum(axis=0, keepdims=True))
        x1 = x1[:,1:] * x1[:,:-1]
        x1 = x1.sum(axis=0)
        x1 = x1.compute()
        x3['gdc_cor'] = np.hstack([x1, [0]])

        x1 = x3.arm.to_numpy()
        x1 = np.hstack([True, x1[1:] != x1[:-1]])
        x2 = np.hstack([True, (x3.dm_cor<0.3)[:-1]])
        x3['g1'] = np.cumsum(x1 | x2)

        x3['dm_n'] = x3.groupby('g1').dm.transform(lambda x: sum(x>0.1))
        x3['gdc_n'] = x3.groupby('g1').gdc.transform(lambda x: sum(x>0.1))
        x3['t'] = (x3.gdc>0.1) & (x3.dm>0.1)
        x3['both_n'] = x3.groupby('g1').t.transform(lambda x: sum(x>0.1))

        x4 = x3.query('both_n>0').copy()
        x4['g2'] = x4.g1.astype('category').cat.codes+1
        x4 = x4.groupby('g2').apply(lambda x: x.loc[x.index[x.t][0]:(x.index[x.t][-1]),:])

        return x4.set_index('cols').g2.astype(int)

        x4.query('g2==28 & t==True')

        np.corrcoef(x3.dm, x3.gdc)
        x3['g1'] = x3.g.astype(str)
        px.scatter(
            x3,
            x='dm', y='gdc', color='g1',
            hover_data=['cols']
        ).show()
    _playground11.dm_cnv_fft_group = dm_cnv_fft_group
    #del self.__lazy__dm_cnv_fft_group

    @lazy_property
    def dm_cnv_fft_svd(self):
        s = Path(self.storage.path)/'dm_cnv_fft_svd'
        if not s.exists():
            fft = self.dm_cnv_fft.copy()
            fft['fft_group'] = self.dm_cnv_fft_group
            svd = _svd1(fft, 10, 'cols')
            for i, x in svd.items():
                x.xarray.to_zarr(s/str(i))
        svd = {
            int(i.name):SVD.from_xarray(xa.open_zarr(s/i))
            for i in s.glob('*')
        }
        return svd
    _playground11.dm_cnv_fft_svd = dm_cnv_fft_svd
    # del self.__lazy__dm_cnv_fft_svd

    @lazy_property
    def dm_cnv_fft_svd_rand(self):
        fft = self.dm_cnv_fft.copy()
        fft['fft_group'] = self.dm_cnv_fft_group
        return _svd1(fft, 10, 'cols', True)
    _playground11.dm_cnv_fft_svd_rand = dm_cnv_fft_svd_rand
    # del self.__lazy__dm_cnv_fft_svd_rand

    @lazy_property
    @cached_property(type=Dir.pickle)
    def dm_cnv_fft_svd_pc(self):
        m = self
        x1 = m.dm_cnv_fft_svd
        x1 = [x.ve.data for x in x1.values()]
        x1 = daa.hstack(x1)
        x1 = x1.compute()
        x4 = m.dm_cnv_fft_svd_rand
        x4 = [x4[i].ve.data for i in m.dm_cnv_fft_svd.keys()]
        x4 = daa.hstack(x4)
        x4 = x4.compute()
        x2 = pd.DataFrame(dict(
            fft_group_pc = np.hstack([np.repeat(i, x.s.shape[0]) for i, x in m.dm_cnv_fft_svd.items()]).astype(int),
            pc = np.hstack([np.arange(x.s.shape[0]) for _, x in m.dm_cnv_fft_svd.items()]).astype(int),
            n = np.hstack([np.repeat(x.s.shape[0], x.s.shape[0]) for _, x in m.dm_cnv_fft_svd.items()]),
            ve = x1,
            ve_rand = x4
        ))
        x2 = x2.query('n==1 | ve_rand-ve>=1e-4').groupby('fft_group_pc').first()
        return x2.pc+1
    _playground11.dm_cnv_fft_svd_pc = dm_cnv_fft_svd_pc
    # del self.__lazy__dm_cnv_fft_svd_pc

    @property
    def dm_expr1(self):
        x1 = self.dm_cnv_fft_svd[0]
        x1 = x1.u[:,:int(self.dm_cnv_fft_svd_pc[0].item())]
        x6 = self.dm_expr.data
        x6 = x6 - x1 @ (x1.T @ x6)
        x6 = dmlp.StandardScaler().fit_transform(x6)
        x6 = x6.persist()
        return x6
    _playground11.dm_expr1 = dm_expr1

    @lazy_property
    def dm_expr_svd(self):
        s = Path(self.storage.path)/'dm_expr_svd'
        if not s.exists():
            x6 = self.dm_expr1
            svd = SVD.from_mat(x6)
            svd.xarray.to_zarr(s)
        svd = SVD.from_xarray(xa.open_zarr(s))
        return svd
    _playground11.dm_expr_svd = dm_expr_svd
    # del self.__lazy__dm_expr_svd

    @lazy_property
    def dm_expr_svd_rand(self):
        x6 = self.dm_expr1
        x6.data = daa.apply_along_axis(np.random.permutation, 0, x6.data, shape=(x6.shape[0],), dtype=x6.dtype)
        return SVD.from_mat(x6)
    _playground11.dm_expr_svd_rand = dm_expr_svd_rand
    # del self.__lazy__dm_expr_svd_rand

    @lazy_property
    @cached_property(type=Dir.pickle)
    def dm_expr_svd_pc(self):
        m = self
        x1 =  m.dm_expr_svd.ve.compute()
        x4 = m.dm_expr_svd_rand.ve.compute()
        x2 = pd.DataFrame(dict(
            pc = np.arange(m.dm_expr_svd.s.shape[0]),
            ve = x1,
            ve_rand = x4
        ))
        x2 = x2.query('ve_rand-ve>=1e-4').pc.iloc[0]
        return x2+1
    _playground11.dm_expr_svd_pc = dm_expr_svd_pc
    # del self.__lazy__dm_expr_svd_pc

    @lazy_property
    def crispr_cnv_fit(self):
        s = Path(self.storage.path)/'crispr_cnv_fit'
        if not s.exists():
            x1 = self.crispr.data.sel(rows=self.dm_cnv.rows.values).data.persist()
            x2 = [x.u[:,:int(self.dm_cnv_fft_svd_pc[int(i)].item())].data for i, x in self.dm_cnv_fft_svd.items()]
            x2 = [((x1 - x @ (x.T @ x1))**2).mean(axis=0) for x in x2]
            x2 = daa.stack(x2).persist()
            x2 = x2.rechunk((-1, None))
            x4 = [x.perm.u[:,:int(self.dm_cnv_fft_svd_pc[int(i)].item())].data for i, x in self.dm_cnv_fft_svd.items()]
            x4 = [((x1 - x @ (x.T @ x1))**2).mean(axis=0) for x in x4]
            x4 = daa.stack(x4).persist()
            x4 = x4.rechunk((-1, None))
            x2.to_zarr(str(s/'obs'))
            x4.to_zarr(str(s/'rand'))

        x2 = daa.from_zarr(str(s/'obs'))
        x4 = daa.from_zarr(str(s/'rand'))

        x5 = dmlp.StandardScaler().fit(daa.log10(1-x4+1e-5).T)
        x6 = x5.transform(daa.log10(1-x2+1e-5).T).T.persist()
        m = xa.Dataset()
        m['fft_group'] = ('fft_group', np.arange(x2.shape[0]))
        m['cols'] = self.crispr.cols
        m['rand_mean'] = ('fft_group', x5.mean_)
        m['rand_var'] = ('fft_group', x5.var_)
        m['r2'] = (('fft_group', 'cols'), x6)
        return m
    _playground11.crispr_cnv_fit = crispr_cnv_fit
    # del self.__lazy__crispr_cnv_fit

    @lazy_property
    def crispr_expr_fit(self):
        return _fit(self.crispr.data, self.dm_expr.data, 400)
    _playground11.crispr_expr_fit = crispr_expr_fit
    # del self.__lazy__crispr_expr_fit

    @lazy_property
    def crispr_model(self):
        def _build():
            x1 = self.crispr_cnv_fit.r2[1:,:]
            x1 = x1.assign_coords(group=('fft_group', ['cnv:'+str(x) for x in x1.fft_group.values]))
            x1 = x1.swap_dims(fft_group='group')
            crispr_cnv_fit_r2 = x1>2

            x1 = crispr_cnv_fit_r2.sum(axis=1).swap_dims(group='fft_group').to_series()
            x1 = x1.pipe(lambda x: x[x > 0])
            x1 = [0] + list(x1.index)
            x1 = {i: self.dm_cnv_fft_svd[i] for i in x1}
            x1 = {i: x.cut(np.s_[:self.dm_cnv_fft_svd_pc[i].item()]) for i, x in x1.items()}
            x1 = {i: x.inv(0) for i, x in x1.items()}
            x1 = {i: xa.merge([x.us.rename('us'), x.v.rename('v')]) for i, x in x1.items()}
            x1 = {i: x.persist() for i, x in x1.items()}
            x1 = {i: x.assign_coords(pc=['cnv:'+str(i)+':'+str(pc) for pc in x.pc.values]) for i, x in x1.items()}
            x1 = {i: x.assign_coords(group=('pc', np.repeat('cnv:' + str(i), x.pc.shape[0]))) for i, x in x1.items()}
            dm_cnv_fft_svd = x1

            x1 = self.dm_expr_svd
            x1 = x1.cut(np.s_[:self.dm_expr_svd_pc])
            x1 = x1.inv(0)
            x1 = xa.merge([x1.us.rename('us'), x1.v.rename('v')])
            x1 = x1.persist()
            x1 = x1.assign_coords(pc = ['expr:'+str(pc) for pc in x1.pc.values])
            x1 = x1.assign_coords(group=('pc', np.repeat('expr' , x1.pc.shape[0])))
            dm_expr_svd = x1

            crispr = self.crispr.data.copy()
            crispr.data = crispr.data.persist()
            crispr = crispr.rename(rows='_rows')

            v = [x.v for i, x in dm_cnv_fft_svd.items()]
            v = v + [dm_expr_svd.v]
            v = xa.concat(v, 'pc')
            v = v.rename(rows='_rows', pc='rows')
            v = v.persist()

            x1 = xa.DataArray(
                daa.full((2, crispr_cnv_fit_r2.shape[1]), True, dtype=bool),
                dims=('group', 'cols'),
                coords=dict(group=['cnv:0', 'expr'], cols=crispr_cnv_fit_r2.cols)
            )
            x1 = xa.concat([x1, crispr_cnv_fit_r2.drop('fft_group')], 'group')
            x1 = x1.rename('r2').to_series()
            x1 = x1.pipe(lambda x: x[x])
            x1 = x1.reset_index().groupby('cols').group.apply(list).to_dict()
            loc_groups = x1

            def model(x, y):
                svd = SVD.from_mat(x).inv(0)
                return svd.u @ (svd.s * (svd.v @ y))

            x1 = [model(
                v.sel(rows=v.group.isin(loc_groups[x])),
                crispr.loc[:, x]
            ) for x in crispr.cols.values]

            x2 = [set(x.rows.values) for x in x1]
            x2 = set().union(*x2)
            x2 = pd.Series(range(len(x2)), index=sorted(x2))

            x3 = [x.cols.item() for x in x1]
            x3 = pd.Series(range(len(x3)), index=x3)

            import sparse

            x4 = daa.hstack([x.data for x in x1]).compute()
            x4 = sparse.COO(
                [
                    x2[np.hstack([x.rows.values for x in x1])],
                    x3[np.hstack([np.repeat(x.cols.values, x.rows.shape[0]) for x in x1])]
                ],
                x4,
                shape=(len(x2), len(x3))
            )
            x4 = x4.astype('float32')
            x4 = xa.DataArray(
                daa.from_array(x4),
                dims=('pc', 'cols'),
                coords={
                    'pc': x2.index,
                    'cols': x3.index
                }
            )

            x7 = {
                'cnv:' + str(i): x.us.drop('fft_group').rename(cols='rows')
                for i, x in dm_cnv_fft_svd.items()
            }
            x7['expr'] = dm_expr_svd.us.rename(cols='rows')

            x8 = namespace()
            x8.u = x4
            x8.vs = x7
            x8.crispr_stats = self.crispr[['mean', 'var']]
            x8.expr_stats = self.dm_expr[['mean', 'var']]
            x8.cnv_stats = self.dm_cnv[['mean', 'var', 'chrom', 'arm', 'txMid']]
            x8.cnv_stats = x8.cnv_stats.merge(self.dm_cnv_fft_stats)

            return x8

        s = Path(self.storage.path)/'crispr_model'
        if not s.exists():
            x8 = _build()
            x1 = x8.u.astype('float16')
            x1.data = daa.from_array(x1.data.compute().todense())
            x1.rename('data').to_dataset().to_zarr(s/'u')
            for i, x in x8.vs.items():
                x1 = x.astype('float16')
                x1['rows'] = x1.rows.astype(str)
                x1.rename('data').to_dataset().to_zarr(s / 'vs' /i)
            x8.crispr_stats.to_zarr(s/'crispr_stats')
            x8.expr_stats.to_zarr(s / 'expr_stats')
            x8.cnv_stats.to_zarr(s / 'cnv_stats')

        x8 = namespace()
        x8.u = xa.open_zarr(s/'u').data.astype('float32').rename('u')
        x8.vs = {
            i.name: xa.open_zarr(i).astype('float32').data.rename(i.name)
            for i in (s/'vs').glob('*')
        }
        x8.crispr_stats = xa.open_zarr(s/'crispr_stats')
        x8.expr_stats = xa.open_zarr(s / 'expr_stats')
        x8.cnv_stats = xa.open_zarr(s / 'cnv_stats')
        return x8
    _playground11.crispr_model = crispr_model
    # del self.__lazy__crispr_model

    def crispr_predict(self, expr, cnv):
        model = self.crispr_model

        expr = expr - model.expr_stats['mean']
        expr = expr/np.sqrt(model.expr_stats['var'])
        expr = expr.rename(rows='_rows', cols='rows')
        expr.data = expr.data.rechunk((None, -1))

        cnv = cnv - model.cnv_stats['mean']
        cnv = cnv/np.sqrt(model.cnv_stats['var'])
        cnv = xa.merge([cnv.rename('data'), model.cnv_stats[['arm', 'chrom', 'txMid']]])
        cnv = cnv.sortby(['chrom', 'txMid'])
        cnv = cnv.rename(rows='_rows')
        cnv.data.data = cnv.data.data.rechunk((None, -1))

        d = cnv.data.assign_coords(arm=cnv.arm)
        d = _fft1(d, 10, 'arm', 'cols')
        d = d.drop('arm')
        d['fft'] = d.fft - model.cnv_stats['fft_mean']
        d['fft'] = d.fft/np.sqrt(model.cnv_stats['fft_var'])
        d['fft_resid'] = d.fft_resid - model.cnv_stats['fft_resid_mean']
        d['fft_resid'] = d.fft_resid/np.sqrt(model.cnv_stats['fft_resid_var'])
        cnv = cnv.merge(d)

        expr_v = expr @ model.vs['expr']
        cnv_v_glob = cnv.fft.rename(freq='rows') @ model.vs['cnv:0']
        cnv_v_loc = [
            cnv.fft_resid.rename(cols='rows') @ x
            for i, x in model.vs.items()
            if i not in ['cnv:0', 'expr']
        ]
        crispr = [expr_v, cnv_v_glob] + cnv_v_loc
        crispr = xa.concat(crispr, 'pc')
        crispr.data = crispr.data.rechunk((None, -1))
        crispr = crispr @ model.u
        crispr = crispr.rename(_rows='rows')
        crispr = crispr * np.sqrt(model.crispr_stats['var'])
        crispr = crispr + model.crispr_stats['mean']

        return crispr
    _playground11.crispr_predict = crispr_predict

    @lazy_property
    def gdc_prediction(self):
        s = Path(self.storage.path) / 'gdc_prediction'

        if not s.exists():
            expr = merge.gdc_expr.data.copy().astype('float32')
            cnv = merge.gdc_cnv.data.copy().astype('float32')
            crispr_predict = self.crispr_predict(expr, cnv)
            crispr_predict = crispr_predict.astype('float16')
            crispr_predict['rows'] = crispr_predict.rows.astype(str)
            crispr_predict['cols'] = crispr_predict.cols.astype(str)
            crispr_predict.data = crispr_predict.data.rechunk((1000, -1))
            crispr_predict.rename('data').to_dataset().to_zarr(s)

        crispr_predict = xa.open_zarr(s).data.astype('float32')
        return crispr_predict
    _playground11.gdc_prediction = gdc_prediction
    # del self.__lazy__gdc_prediction

    @lazy_property
    def dm_prediction(self):
        s = Path(self.storage.path) / 'dm_prediction'

        if not s.exists():
            expr = merge.dm_expr.data.copy().astype('float32')
            cnv = merge.dm_cnv.data.copy().astype('float32')
            crispr_predict = self.crispr_predict(expr, cnv)
            crispr_predict = crispr_predict.astype('float16')
            crispr_predict['rows'] = crispr_predict.rows.astype(str)
            crispr_predict['cols'] = crispr_predict.cols.astype(str)
            crispr_predict.data = crispr_predict.data.rechunk((1000, -1))
            crispr_predict.rename('data').to_dataset().to_zarr(s)

        crispr_predict = xa.open_zarr(s).data.astype('float32')
        return crispr_predict
    _playground11.dm_prediction = dm_prediction
    # del self.__lazy__dm_prediction

    @lazy_property
    def crispr_model_score(self):
        x1 = self.dm_prediction.copy()
        x1['train'] = self.train_split.train
        x1 = x1.groupby('train').apply(lambda x: x-x.mean(axis=0))
        x1 = x1.groupby('train').apply(lambda x: x/np.sqrt((x**2).sum(axis=0)))

        x2 = merge.crispr.data.copy()
        x2['train'] = self.train_split.train
        x2 = x2.groupby('train').apply(lambda x: x-x.mean(axis=0))
        x2 = x2.groupby('train').apply(lambda x: x/np.sqrt((x**2).sum(axis=0)))

        x3 = x1*x2
        x3 = x3.groupby('train').sum(dim='rows')

        x3 = x3.to_dataframe().reset_index().pivot_table(index='cols', columns='train', values='data')
        x3['n'] = (np.abs(self.crispr_model.u)>0).sum(axis=0).to_series()
        x3['m'] = self.train_split.train.sum().item()

        return x3
    _playground11.crispr_model_score = crispr_model_score
    # del self.__lazy__crispr_model_score

    @lazy_property
    @cached_property(type=Dir.pickle)
    def dm_rows_annot(self):
        x1 = merge.dm_expr1.drop_dims('cols')
        x2 = merge.dm_cnv1.drop_dims(['cols', 'cyto_cols'])
        x3 = merge.crispr1.drop_dims('cols')
        return xa.merge([x1, x2, x3])
    _playground11.dm_rows_annot = dm_rows_annot
    # del self.__lazy__dm_rows_annot

    @lazy_property
    @cached_property(type=Dir.pickle)
    def gdc_rows_annot(self):
        x1 = merge.gdc_expr1.drop_dims('cols').drop('is_normal')
        x2 = merge.gdc_cnv1.drop_dims(['cols', 'cyto_cols'])
        return xa.merge([x1, x2])
    _playground11.gdc_rows_annot = gdc_rows_annot
    # del self.__lazy__gdc_rows_annot

    @lazy_property
    def crispr_prediction(self):
        x1 = self.dm_prediction.copy()
        x1['row_label'] = self.dm_rows_annot.stripped_cell_line_name
        x1['source'] = self.dm_rows_annot.lineage_subtype
        x1['source'] = ('rows', ['CCLE-' + x for x in x1.source.astype(str).values])
        x1['train'] = self.train_split.train
        x1['observed'] = ('rows', np.repeat(False, x1.rows.shape[0]))
        x1['dataset'] = ('rows', np.repeat('DepMap', x1.rows.shape[0]))

        x2 = self.gdc_prediction.copy()
        x2['row_label'] = self.gdc_rows_annot.case_id
        x2['source'] = self.gdc_rows_annot.project_id
        x2['train'] = ('rows', np.repeat(False, x2.rows.shape[0]))
        x2['observed'] = ('rows', np.repeat(False, x2.rows.shape[0]))
        x2['dataset'] = ('rows', np.repeat('TCGA', x2.rows.shape[0]))

        x3 = merge.crispr.data.copy().astype('float32')
        x3['row_label'] = self.dm_rows_annot.stripped_cell_line_name
        x3['source'] = self.dm_rows_annot.lineage_subtype
        x3['source'] = ('rows', ['CCLE-' + x for x in x3.source.astype(str).values])
        x3['train'] = self.train_split.train
        x3['observed'] = ('rows', np.repeat(True, x3.rows.shape[0]))
        x3['dataset'] = ('rows', np.repeat('DepMap', x3.rows.shape[0]))

        crispr = xa.concat([x1, x2, x3], 'rows')
        crispr = crispr.assign_coords(merge.crispr[['symbol', 'entrez']])
        return crispr
    _playground11.crispr_prediction = crispr_prediction
    # del self.__lazy__crispr_prediction

    def predict(self, col = None, symbol = None, entrez = None):
        if col is None:
            if entrez is None:
                if symbol is None:
                    raise(ValueError('col or symbol or entrez'))
                else:
                    x5 = self.crispr.cols.sel(cols=self.crispr.symbol == symbol).item()
            else:
                x5 = self.crispr.cols.sel(cols=self.crispr.entrez == entrez).item()
        else:
            x5 = col

        x1 = self.crispr_cnv_fit.r2.loc[1:,x5].to_series().pipe(lambda x: x[x>2]).index
        x1 = [0] + list(x1)
        x1 = [x.u[:,:int(self.dm_cnv_fft_svd_pc[int(i)].item())] for i, x in self.dm_cnv_fft_svd.items() if i in x1]
        x1 = x1 + [self.dm_expr_svd.u[:,:self.dm_expr_svd_pc]]
        x1 = [x.assign_coords(pc=([str(i)+':'+x for x in x.pc.values.astype(str)])) for i, x in enumerate(x1)]
        x1 = xa.concat(x1, 'pc').rename({'pc': 'cols'})
        x1 = SVD.from_mat(x1).persist()
        x2 = self.crispr.data.loc[:,x5].compute()
        x3 = x1.u
        x3 = (x3 @ (x3.T @ x2)).compute()
        return xa.merge([
            x2.rename('obs'),
            x3.rename('pred'),
            (1 - (x2 - x3) ** 2).mean().rename('r2'),
            xa.DataArray(
                x1.u.shape[1] / x1.u.shape[0],
                dims=(),
                coords={'cols': x2.cols},
                name='r2_rand'
            )
        ])
    _playground11.predict = predict
_()

# %%
def _():
    for i in range(5):
        print(i)
        self = _playground11(f'20230509/0.8/{i}', 0.8)
        Path(self.storage.path).mkdir(
            parents=True, exist_ok=True
        )
        _ = self.dm_prediction
    
# %%
def _():
    self = _playground11(f'20230509/0.8/1', 0.8)
    Path(self.storage.path).mkdir(
        parents=True, exist_ok=True
    )

    self.train_split
    self.crispr
    self.dm_expr
    self.dm_cnv
    self.gdc_cnv

    self.crispr_model

# %%
def _():
    import string_db
    x1 = self.dm_expr.assign_coords(self.dm_cnv[['symbol']]).data
    #x1_1 = x1.cols.values
    x1_1 = np.random.choice(x1.cols.values, 1000)
    x1 = x1.sel(cols=x1_1)
    x6 = np.repeat('?', x1.shape[1])
    x6 = ','.join(x6)
    x6 = string_db.query(f'select preferred_name, protein_external_id from info where preferred_name in ({x6})', x1.symbol.values)
    x3 = np.repeat('?', x6.shape[0])
    x3 = ','.join(x3)
    x3 = string_db.query(
        f'select * from links where protein1 in ({x3}) and protein2 in ({x3})',
        np.hstack([x6.protein_external_id, x6.protein_external_id])
    )
    x6 = x6[x6.protein_external_id.isin(np.unique(x3[['protein1', 'protein2']].to_numpy().ravel()))]
    x6 = x6.rename(columns={'preferred_name': 'symbol'}).set_index('symbol').protein_external_id
    x6 = x6.drop_duplicates()
    x1 = x1.sel(cols=x1.symbol.isin(x6.index))
    x1 = x1.swap_dims(cols='symbol')
    x1 = x1[:, x1.symbol.to_series().reset_index(drop=True).drop_duplicates().index]
    x1['protein'] = x6
    x1 = x1.swap_dims(symbol='protein')
    x1 = x1.drop(['cols', 'symbol'])
    x1 = x1/np.sqrt((x1**2).sum(dim='rows'))
    #x1.data = daa.apply_along_axis(np.random.permutation, 0, x1.data, shape=(x1.shape[0],)).persist()
    x1 = x1.rename(protein='protein1').T @ x1.rename(protein='protein2')
    x1 = x1.persist()
    x2 = SVD.from_mat(x1, solver='full')
    x2.s = 1/(x2.s+1)
    x2 = x2.usv.persist()
    #x2 = x2.cut(np.s_[:400]).inv(0).usv.persist()
    x2_3 = np.diag(np.sqrt(1/np.diag(x2)))
    x2.data = x2_3 @ x2.data @ x2_3
    x2 = x2.persist()
    x3['score'] = x3.experiments.astype(int)
    x4 = pd.pivot_table(x3, values='score', index='protein1', columns='protein2', fill_value=0)
    x4 = xa.DataArray(x4)
    x5 = xa.merge([x1.rename('data'), x2.rename('inv'), x4.rename('score')])
    plt.hist((x5.data - np.diag(np.diag(x5.data))).values.ravel(), 100)
    plt.hist((x5.inv - np.diag(np.diag(x5.inv))).values.ravel(), 100)
    pd.Series(x5.score.values.ravel()).pipe(lambda x: x[x > 0]).sort_values()
    x5.to_dataframe().query('score>0').plot.scatter('data', 'score')
    np.corrcoef(
        x5.inv.values.ravel()**2,
        (x5.score.values.ravel())
    )
    pd.crosstab(
        x5.data.values.ravel()>0.4,
        np.random.permutation(x5.score.values.ravel()>600),
        margins=False
    )

    import scipy.cluster.hierarchy as spch
    import scipy.spatial.distance as spsd

    x7 = spch.linkage(spsd.squareform(1-x1, checks=False), method='average')
    x7 = spch.to_tree(x7).pre_order()
    spch.dendrogram(x7)
    plt.imshow((x1 - np.diag(np.diag(x1)))[x7, x7], cmap='bwr', vmin=-0.6, vmax=0.6)

    x7 = spch.linkage(x1.T, method='average')
    x7 = spch.to_tree(x7).pre_order()
    spch.dendrogram(x7)
    plt.imshow((x1 - np.diag(np.diag(x1)))[x7, x7], cmap='bwr', vmin=-0.6, vmax=0.6)

    x8 = spch.linkage(spsd.squareform(1-x2, checks=False), method='average')
    x8 = spch.to_tree(x8).pre_order()
    x9 = x2[x8,x8]
    plt.imshow((x9 - np.diag(np.diag(x9))), cmap='bwr', vmin=-0.1, vmax=0.1)
    spch.dendrogram(x8)
    plt.hist(x2.values.ravel(), 1000)


    self.crispr_model_score.sort_values(True).tail(20)

    px.scatter(
        self.crispr_model_score.reset_index().rename(columns={False: 'test', True: 'train'}),
        'train', 'test',
        hover_data=['cols']
    ).show()

    crispr = self.crispr_prediction

    plot_data = crispr.sel(cols=crispr.symbol=='WRN').squeeze().to_dataframe().sort_values(['source'])
    plot_data['color'] = plot_data.dataset+','+np.where(plot_data.observed, 'obs', 'pred')+','+np.where(plot_data.train, 'train', 'test')
    px.scatter(
        plot_data,
        x='source', y='data',
        color='color',
        hover_data=['row_label'],
        title=plot_data.symbol[0]
    ).show()

    self = playground11
    self.dm_cnv_fft_svd_pc

    m = playground11
    plot_fft_resid(m.dm_cnv.merge(m.dm_cnv_fft).drop('mean'))
    plot_fft(m.dm_cnv.merge(m.dm_cnv_fft).drop('mean'))

    m = playground11
    (m.crispr_cnv_fit.r2[1:,:]>3).sum(axis=1).to_series().sort_values()


    m = playground11
    x5 = m.predict(symbol='WRN')
    plt.plot(x5.obs, x5.pred, '.')
    print(x5.r2.round(2).item(), x5.r2_rand.round(2).item())


    m = playground11
    plt.figure()
    plt.gca().plot(m.crispr_cnv_fit.r2[0,:], m.crispr_expr_fit.r2, '.')

    plt.figure()
    plt.gca().plot(m.crispr_cnv_fit.r2[0,:], m.crispr_cnv_fit.r2[1:,:].max(axis=0), '.')

    plt.figure()
    plt.gca().plot(m.crispr_expr_fit.r2, m.crispr_cnv_fit.r2[1:,:].mean(axis=0), '.')

    plt.figure()
    plt.gca().plot(m.crispr_cnv_fit.r2[0,:], m.crispr_cnv_fit.r2[1,:], '.')

    pd.DataFrame(dict(
        expr=(m.crispr_expr_fit.r2 > 2).values,
        cnv_glob = (m.crispr_cnv_fit.r2[0,:]>2).values,
        cnv_loc = ((m.crispr_cnv_fit.r2[1:,:]>2).sum(axis=0)>2).values
    )).value_counts().sort_index()#.reset_index().pivot(index='loc', columns='glob')

# %%
def _():
    x1 = (2*np.random.rand(9)-1).reshape((3,3))
    x1 = (x1+x1.T)/2
    x1 -= np.diag(np.diag(x1))
    x1 += np.eye(len(x1))
    print(
        x1, 
        np.linalg.eigvalsh(x1), 
        sep='\n'
    )
    x1 = np.linalg.cholesky(x1)
    x2 = np.random.randn(100*len(x1)).reshape((-1,len(x1)))
    x2 -= x2.mean(axis=0)
    x2, _ = np.linalg.qr(x2)
    x2 = x2 @ x1.T
    print(x2.T @ x2)

# %%
class Model1:
    def fit(self, train):
        crispr, expr, cnv = [
            _scale1(x).persist().rename('data').reset_coords(['center', 'scale'])
            for x in [train.crispr, train.expr, train.cnv]
        ]

        cnv_svd = [
            (a, SVD.from_mat(x.data).inv())
            for a, x in cnv.groupby('arm')
        ]
        cnv_svd = {
            a: xa.merge([x.v.rename('u'), x.us.rename('vs')]).\
            pipe(lambda x: x.sel(pc=range(5))).\
            persist().\
            assign(
                arm=lambda x: ('pc', [a]*x.sizes['pc']),
                arm_pc=lambda x: ('pc', a+':'+x.pc.astype(str).to_series())
            ).set_coords('arm').swap_dims(pc='arm_pc') 
            for a, x in cnv_svd
        }
        cnv_svd = namespace(
            u=xa.concat([x.u for x in cnv_svd.values()], dim='arm_pc'),
            vs={a: x.vs for a, x in cnv_svd.items()}
        )
        self.cnv_svd = cnv_svd
    
        cnv_svd1 = SVD.from_mat(cnv_svd.u).inv()
        cnv_svd1 = xa.merge([cnv_svd1.v.rename('u'), cnv_svd1.us.rename('vs')]).persist()
        cnv_svd1 = cnv_svd1.assign(
            src=lambda x: ('pc', ['cnv']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['cnv:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        self.cnv_svd1 = cnv_svd1

        expr1_svd = expr.data - cnv_svd1.u @ (cnv_svd1.u @ expr.data)
        expr1_svd = SVD.from_mat(expr1_svd).inv()
        expr1_svd = xa.merge([expr1_svd.v.rename('u'), expr1_svd.us.rename('vs')]).persist()
        expr1_svd = expr1_svd.sel(pc=range(100))
        expr1_svd = expr1_svd.assign(
            src=lambda x: ('pc', ['expr']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['expr:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        self.expr1_svd = expr1_svd

        u = xa.concat([cnv_svd1.u, expr1_svd.u], dim='src_pc')

        self.proj = (u @ crispr.data).persist()
        self.proj1 = (cnv_svd1.u @ expr.data).persist()

        self.expr = expr.drop('data')
        self.cnv = cnv.drop('data')
        self.crispr = crispr.drop('data')

        return self
    
    @compose(property, lazy)
    def coef(self):                
        coef1 = self.proj @ self.cnv_svd1.vs

        coef2 = xa.concat([
            coef1 @ x
            for x in self.cnv_svd.vs.values()
        ], dim='cnv_cols')

        coef3 = self.proj @ self.expr1_svd.vs

        coef4 = self.proj1 @ self.cnv_svd1.vs

        coef5 = xa.concat([
            coef4 @ x
            for x in self.cnv_svd.vs.values()
        ], dim='cnv_cols')

        return (coef1, coef2, coef3, coef4, coef5)

    def cnv_cor_plot(self, x):
        cnv_cor = self.cnv_svd.u @ _scale1(x)
        cnv_cor['col_arm'] = self.cnv.arm.rename(cnv_cols='cols').drop('arm')
        cnv_cor = (cnv_cor**2).compute().rename('r2')
        cnv_cor = cnv_cor.to_dataframe().sort_values('r2')
        cnv_cor['f'] = cnv_cor.arm==cnv_cor.col_arm
        sns.boxplot(
            x='f',
            y='r2',
            data=cnv_cor[cnv_cor.pc==0]
        )

    def cnv_cor_plot1(self, x):
        cnv_cor = self.cnv_svd1.u @ _scale1(x)
        cnv_cor = (cnv_cor**2).rename('r2').to_dataframe().reset_index()
        cnv_cor = cnv_cor.groupby('cols').r2.sum()
        print(cnv_cor.mean())
        sns.histplot(
            x='r2',
            data=cnv_cor.to_frame()
        )
        plt.show()

    def expr_cor(self, x):
        expr_cor = self.expr1_svd.u @ _scale1(x)
        expr_cor = (expr_cor**2).rename('r2').to_dataframe().reset_index()
        expr_cor = expr_cor.groupby('cols').r2.sum()
        print(expr_cor.mean())
        sns.histplot(
            x='r2',
            data=expr_cor.to_frame()
        )
        plt.show()
    
    def predict(self, test):
        cnv1 = ((test.cnv-self.cnv.center)/self.cnv.scale).persist()
        expr1 = (test.expr-self.expr.center)/self.expr.scale
        expr1 = (expr1 - self.coef[4] @ cnv1).persist()

        crispr3 = xa.Dataset()
        crispr3['cnv'] = self.coef[1] @ cnv1
        crispr3['expr'] = self.coef[2] @ expr1
        crispr3['pred'] = self.crispr.scale * (crispr3.cnv + crispr3.expr) + self.crispr.center
        return crispr3
    
    @classmethod
    def _from_dict(cls, **elems):
        self = cls()
        for k, v in elems.items():
            setattr(self, k, v)
        return self

    def store(self, storage):
        self.cache.store(self, storage)

    @classmethod
    def restore(cls, storage):
        return cls.cache.restore(storage)
    
Model1.cache = ClassCache(
    Model1._from_dict,
    cnv_svd = ClassCache(
        namespace,
        u = XArrayCache(),
        vs = DictCache(XArrayCache())
    ),
    cnv_svd1 = XArrayCache(),
    expr1_svd = XArrayCache(),
    proj = XArrayCache(),
    proj1 = XArrayCache(),
    expr = XArrayCache(),
    cnv = XArrayCache(),
    crispr = XArrayCache()
)

# %%
class Model2:
    def fit(self, train):
        crispr, expr, cnv = [
            _scale1(x).persist().rename('data').reset_coords(['center', 'scale'])
            for x in [train.crispr, train.expr, train.cnv]
        ]
    
        cnv_svd1 = SVD.from_mat(cnv.data).inv()
        cnv_svd1 = xa.merge([cnv_svd1.v.rename('u'), cnv_svd1.us.rename('vs')])
        cnv_svd1 = cnv_svd1.sel(pc=range(205)).persist()
        cnv_svd1 = cnv_svd1.assign(
            src=lambda x: ('pc', ['cnv']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['cnv:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        self.cnv_svd1 = cnv_svd1

        expr1_svd = expr.data - cnv_svd1.u @ (cnv_svd1.u @ expr.data)
        expr1_svd = SVD.from_mat(expr1_svd).inv()
        expr1_svd = xa.merge([expr1_svd.v.rename('u'), expr1_svd.us.rename('vs')]).persist()
        expr1_svd = expr1_svd.sel(pc=range(100))
        expr1_svd = expr1_svd.assign(
            src=lambda x: ('pc', ['expr']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['expr:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        self.expr1_svd = expr1_svd

        u = xa.concat([cnv_svd1.u, expr1_svd.u], dim='src_pc')

        self.proj = (u @ crispr.data).persist()
        self.proj1 = (cnv_svd1.u @ expr.data).persist()

        self.expr = expr.drop('data')
        self.cnv = cnv.drop('data')
        self.crispr = crispr.drop('data')

        return self
    
    @compose(property, lazy)
    def coef(self):                
        coef1 = self.proj @ self.cnv_svd1.vs
        coef3 = self.proj @ self.expr1_svd.vs
        coef4 = self.proj1 @ self.cnv_svd1.vs

        return (coef1, coef3, coef4)

    def cnv_cor_plot1(self, x):
        cnv_cor = self.cnv_svd1.u @ _scale1(x)
        cnv_cor = (cnv_cor**2).rename('r2').to_dataframe().reset_index()
        cnv_cor = cnv_cor.groupby('cols').r2.sum()
        print(cnv_cor.mean())
        sns.histplot(
            x='r2',
            data=cnv_cor.to_frame()
        )
        plt.show()

    def expr_cor(self, x):
        expr_cor = self.expr1_svd.u @ _scale1(x)
        expr_cor = (expr_cor**2).rename('r2').to_dataframe().reset_index()
        expr_cor = expr_cor.groupby('cols').r2.sum()
        print(expr_cor.mean())
        sns.histplot(
            x='r2',
            data=expr_cor.to_frame()
        )
        plt.show()
    
    def predict(self, test):
        cnv1 = ((test.cnv-self.cnv.center)/self.cnv.scale).persist()
        expr1 = (test.expr-self.expr.center)/self.expr.scale
        expr1 = (expr1 - self.coef[2] @ cnv1).persist()

        crispr3 = xa.Dataset()
        crispr3['cnv'] = self.coef[0] @ cnv1
        crispr3['expr'] = self.coef[1] @ expr1
        crispr3['pred'] = self.crispr.scale * (crispr3.cnv + crispr3.expr) + self.crispr.center
        return crispr3
    
    @classmethod
    def _from_dict(cls, **elems):
        self = cls()
        for k, v in elems.items():
            setattr(self, k, v)
        return self

    def store(self, storage):
        self.cache.store(self, storage)

    @classmethod
    def restore(cls, storage):
        return cls.cache.restore(storage)
    
Model2.cache = ClassCache(
    Model2._from_dict,
    cnv_svd1 = XArrayCache(),
    expr1_svd = XArrayCache(),
    proj = XArrayCache(),
    proj1 = XArrayCache(),
    expr = XArrayCache(),
    cnv = XArrayCache(),
    crispr = XArrayCache()
)

# %%
def gmm(x1, k):
    from sklearn.mixture import GaussianMixture

    d = [x1[d] for d in x1.dims]
    x2 = GaussianMixture(k, verbose=2).fit(x1.data)
    x3 = xa.Dataset()
    x3['m'] = xa.DataArray(x2.means_, [('clust', range(x2.n_components)), d[1]])
    x3['p'] = xa.DataArray(x2.predict_proba(x1.data), [d[0], x3.clust])

    u = x1-x3.m
    w = x3.p/x3.p.sum(dim=d[0].name)
    u = np.sqrt(w) * u
    u = u.transpose('clust', *x1.dims)
    u, s, vt = np.linalg.svd(u.values, full_matrices=False)
    x3['s'] = xa.DataArray(s, [x3.clust, ('pc', range(s.shape[1]))])
    x3['u'] = xa.DataArray(u, [x3.clust, d[0], x3.pc])
    x3['v'] = xa.DataArray(vt, [x3.clust, x3.pc, d[1]])

    return x3

def log_proba(x, vs, cols, pcs):
    p1 = xa.dot(x, vs, dims=cols)
    p2 = np.log(2*np.pi)*vs.sizes[cols]
    p3 = (vs**2).sum(dim=cols)
    p3 = np.log(p3).sum(dim=pcs)
    p1 = (p1**2).sum(dim=pcs)
    p1 = -0.5*(p2-p3+p1)
    return p1

# %%
class Model3:
    def fit(self, train):
        # %%
        crispr, expr, cnv = [
            _scale1(x).rename('data').reset_coords(['center', 'scale'])
            for x in [train.crispr, train.expr, train.cnv]
        ]

        cnv_svd = SVD.from_mat(cnv.data).inv()
        cnv_svd = xa.merge([cnv_svd.v.rename('u'), cnv_svd.us.rename('vs')])
        cnv_svd = cnv_svd.sel(pc=range(205)).persist()
        cnv_svd = cnv_svd.assign(
            src=lambda x: ('pc', ['cnv']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['cnv:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')

        expr_svd = expr.data - cnv_svd.u @ (cnv_svd.u @ expr.data)
        expr_svd = SVD.from_mat(expr_svd).inv()
        expr_svd = xa.merge([expr_svd.v.rename('u'), expr_svd.us.rename('vs')])
        expr_svd = expr_svd.sel(pc=range(100)).persist()
        expr_svd = expr_svd.assign(
            src=lambda x: ('pc', ['expr']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['expr:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')

        # %%
        g = xa.concat([cnv_svd.u, expr_svd.u], dim='src_pc')
        g = gmm(g.transpose('rows', 'src_pc'), 2)
        g['s'] = 1/(g.s+1e-3)
        g['vs'] = g.v*g.s
        k = g.sizes['src_pc']     
        log_det = np.log(g.s).sum(dim='pc')
        g['nf'] = -0.5*np.log(2*np.pi)*k+log_det
        g['pu'] = np.sqrt(g.p)*g.u
        
        self.proj = g.pu @ crispr.data
        self.proj1 = cnv_svd.u @ expr.data
        self.gmm = g.drop(['v', 's', 'u', 'p', 'pu'])
        self.cnv_svd = cnv_svd.drop('u')        
        self.expr_svd = expr_svd.drop('u')
        self.expr = expr.drop('data')
        self.cnv = cnv.drop('data')
        self.crispr = crispr.drop('data')

        # %%

        # %%

        return self
    
    @compose(property, lazy)
    def coef(self):
        # %%
        coef1 = xa.dot(self.proj, self.gmm.vs, dims='pc') @ self.cnv_svd.vs
        coef3 = xa.dot(self.proj, self.gmm.vs, dims='pc') @ self.expr_svd.vs
        coef4 = self.proj1 @ self.cnv_svd.vs
        self.coef = (coef1, coef3, coef4)

        # %%
        return (coef1, coef3, coef4)

    def cnv_cor_plot1(self, x):
        cnv_cor = self.cnv_svd1.u @ _scale1(x)
        cnv_cor = (cnv_cor**2).rename('r2').to_dataframe().reset_index()
        cnv_cor = cnv_cor.groupby('cols').r2.sum()
        print(cnv_cor.mean())
        sns.histplot(
            x='r2',
            data=cnv_cor.to_frame()
        )
        plt.show()

    def expr_cor(self, x):
        expr_cor = self.expr1_svd.u @ _scale1(x)
        expr_cor = (expr_cor**2).rename('r2').to_dataframe().reset_index()
        expr_cor = expr_cor.groupby('cols').r2.sum()
        print(expr_cor.mean())
        sns.histplot(
            x='r2',
            data=expr_cor.to_frame()
        )
        plt.show()
    
    def predict(self, test):
        # %%
        cnv1 = (test.cnv-self.cnv.center)/self.cnv.scale
        cnv1 @= self.cnv_svd.vs

        expr1 = (test.expr-self.expr.center)/self.expr.scale
        expr1 -= self.proj1 @ cnv1
        expr1 @= self.expr_svd.vs

        pred = xa.concat([cnv1, expr1], dim='src_pc')
        pred = pred - self.gmm.m
        pred = xa.dot(pred, self.gmm.vs, dims='src_pc')
        prob = self.gmm.nf - 0.5*(pred**2).sum(dim='pc')
        m = prob.max(dim='clust')
        prob = np.exp(prob - m)
        score = prob.sum(dim='clust')
        prob /= score
        score = m + np.log(score)

        pred = xa.dot(pred, self.proj, dims='pc')
        pred = xa.dot(pred, np.sqrt(prob), dims='clust')
        pred = self.crispr.scale * pred + self.crispr.center

        crispr3 = xa.Dataset()
        crispr3['pred'] = pred
        crispr3['score'] = score

        # %%
        x = xa.merge([crispr3.score, data.train]).to_dataframe()
        (
            p9.ggplot(x)+p9.aes('train', 'score')+p9.geom_boxplot()
        )

        # %%

        return crispr3
    
    @classmethod
    def _from_dict(cls, **elems):
        self = cls()
        for k, v in elems.items():
            setattr(self, k, v)
        return self

    def store(self, storage):
        self.cache.store(self, storage)

    @classmethod
    def restore(cls, storage):
        return cls.cache.restore(storage)
    
Model3.cache = ClassCache(
    Model3._from_dict,
    cnv_svd1 = XArrayCache(),
    expr1_svd = XArrayCache(),
    proj = XArrayCache(),
    proj1 = XArrayCache(),
    expr = XArrayCache(),
    cnv = XArrayCache(),
    crispr = XArrayCache()
)

# %%
def _():    
    pass    

    # %%
    class _analysis2:
        @compose(property, lazy)
        def storage(self):
            return config.cache / 'playground11' / self.name

        def __init__(self, name, train_split_ratio):
            self._train_split_ratio = train_split_ratio
            self.name = name

        @compose(property, lazy, PickleCache(compressor=None))
        def train_split_ratio(self):
            return self._train_split_ratio        

        @compose(property, lazy, PickleCache(compressor=None))
        def train_split(self):
            rows = merge.crispr.rows
            rows['train'] = ('rows', np.random.random(rows.shape[0])<=self.train_split_ratio)
            return rows
        
        @compose(property, lazy)
        def data(self):
            data = xa.merge([
                merge.dm_expr.rename(data='expr', cols='expr_cols'),
                merge.dm_cnv.rename(data='cnv', cols='cnv_cols'),
                merge.crispr.rename(data='crispr', cols='crispr_cols')
            ], join='inner', compat='override')
            data = data.set_coords('arm')
            for x in ['expr', 'cnv', 'crispr']:
                data[x] = data[x].astype(np.float32)
            data['train'] = self.train_split.train
            return data

        @compose(property, lazy, Model1.cache)
        def model1(self):
            train = self.data.sel(rows=self.data.train)
            return Model1().fit(train)
        
        @compose(property, lazy, Model2.cache)
        def model2(self):
            train = self.data.sel(rows=self.data.train)
            return Model2().fit(train)
        
    # %%
    self = _analysis2('20230531/0.8', 0.8)

    # %%
    data = self.data
    train = data.sel(rows=data.train)

    # %%
    model = self.model2

    # %%
    model.cnv_cor_plot(train.expr.rename(expr_cols='cols'))

    # %%
    model.cnv_cor_plot(train.crispr.rename(crispr_cols='cols'))

    # %%    
    model.cnv_cor_plot1(train.expr.rename(expr_cols='cols'))

    # %%
    model.cnv_cor_plot1(train.crispr.rename(crispr_cols='cols'))

    # %%
    model.expr_cor(train.crispr.rename(crispr_cols='cols'))

    # %%
    #model1
    crispr1 = xa.merge([
        model.proj.rename('proj'),
        model.proj1.rename('proj1'),
        model.coef[0].rename('coef1'),
        model.coef[1].rename('coef2'),
        model.coef[2].rename('coef3'),
        model.coef[3].rename('coef4'),
        model.coef[4].rename('coef5')
    ], join='inner')

    # %%
    #model2
    crispr1 = xa.merge([
        model.proj.rename('proj'),
        model.proj1.rename('proj1'),
        model.coef[0].rename('coef1'),
        model.coef[1].rename('coef3'),
        model.coef[2].rename('coef4'),
    ], join='inner')


    # %%
    crispr3 = model.predict(data)
    crispr3['data'] = data.crispr
    crispr3['train'] = data.train
    crispr3 = crispr3.persist()
    
    # %%
    x1 = crispr3.sel(rows=crispr3.train)[['cnv', 'expr']]
    x1 = (x1**2).sum(dim='rows')
    x1 = x1.to_dataframe()
    print(x1.mean())
    print(
        p9.ggplot(x1)+
            p9.aes('cnv', 'expr')+
            p9.geom_point(alpha=0.1, size=2)+
            p9.geom_hline(yintercept=0.23)+
            p9.geom_vline(xintercept=0.40)
    )
    import sklearn.metrics as sklm
    print(
        sklm.confusion_matrix(x1.cnv>0.40, x1.expr>0.23)
    )

    # %%
    x1.query('expr>0.3 & cnv>0.4')

    # %%
    x1 = crispr3[['pred', 'data', 'train']]
    x1 = x1.groupby('train').apply(
        lambda x: x[['pred', 'data']].\
            pipe(lambda x: x-x.mean(dim='rows')).\
            pipe(lambda x: x/np.sqrt((x**2).sum(dim='rows'))).\
            pipe(lambda x: (x.pred * x.data).sum(dim='rows'))
    ).rename('cor')
    x1 = x1.to_dataframe()
    x1 = x1.reset_index().pivot_table(
        index='crispr_cols', columns='train', values='cor'
    )
    x1.columns = x1.columns.astype(str)
    print(
        p9.ggplot(x1)+
            p9.aes('False', 'True')+
            p9.geom_point(alpha=0.1, size=2)
    )

    # %%
    x1.sort_values('False')

    # %%    
    x2 = crispr1.sel(crispr_cols=['ZFYVE16 (9765)']).persist()
    x2 = xa.merge([x2, crispr3], join='inner')

    # %%
    x3 = x2[['data', 'pred', 'train']].to_dataframe()
    print(
        p9.ggplot(x3)+            
            p9.aes('data', 'pred', color='train')+
            p9.geom_point()+
            p9.geom_smooth(method='lm')
    )
    print(
        x3.groupby('train').aggregate(lambda x: x.loc[:, ['data', 'pred']].corr().iloc[0,1])
    )

    # %%
    from scipy.stats import entropy
    x2['coef1'].rename('r').to_dataframe().assign(
        r2=lambda x: x.r**2/(x.r**2).sum()
    ).r2.sort_values().\
        pipe(lambda x: np.exp(entropy(x)))
        #plot(kind='bar')
        #tail(10)    

    # %%
    x2['coef2'].rename('r').to_dataframe().assign(
        r2=lambda x: x.r**2
    ).sort_values('r2').tail(10)

    # %%
    x2['coef3'].rename('r').to_dataframe().assign(
        r2=lambda x: x.r**2
    ).sort_values('r2').tail(10)


# %%
def _():    
    pass    

    # %%
    self = _playground11('full', 1)

    # %%
    crispr = self.crispr.copy()
    crispr['data'] = crispr.data/np.sqrt((crispr.data**2).sum(dim='rows'))
    crispr['data'] = crispr.data.persist()    

    expr = self.dm_expr.copy()
    expr['data'] = expr.data/np.sqrt((expr.data**2).sum(dim='rows'))
    expr['data'] = expr.data.persist()

    cnv = self.dm_cnv.copy()

    # %%
    def inter(x5):
        x6 = daa.concatenate(
            [x5[:,1:], x5[:,[-1]]],
            axis=1
        )
        x6 = daa.concatenate(
            [x5, x5*x5, x5*x6], 
            axis=1
        )
        x6 = x6/np.sqrt((x6**2).sum(axis=0))
        x6 = xa.DataArray(x6, [
            ('rows', x5.rows.data),
            ('cols', list(x5.cols.data)+[x+':1' for x in x5.cols.data]+[x+':2' for x in x5.cols.data])
        ])
        return x6

    cnv_u = [
        SVD.from_mat(inter(x5.data)).u.to_dataset().assign(
            arm=lambda x: ('pc', [a]*x.sizes['pc']),
            arm_pc=lambda x: ('pc', a+':'+x.pc.astype(str).to_series())
        ).set_coords('arm').swap_dims(pc='arm_pc').u
        for a, x5 in cnv.groupby('arm')
    ]
    cnv_u = dask.persist(*cnv_u)
    cnv_u = xa.concat(cnv_u, dim='arm_pc').rename('u')

    # %%
    cnv_cor = cnv_u @ expr.data
    cnv_cor['col_arm'] = cnv.arm
    cnv_cor = (cnv_cor**2).sel(arm_pc=cnv_cor.pc<20).compute().rename('r2')
    cnv_cor = cnv_cor.to_dataframe().sort_values('r2')
    cnv_cor['f'] = cnv_cor.arm==cnv_cor.col_arm

    # %%
    sns.boxplot(
        x='f',
        y='r2',
        data=cnv_cor[cnv_cor.pc==4]
    )

    # %%
    cnv_u1 = cnv_u.sel(arm_pc=cnv_u.pc<5)
    cnv_u1 = SVD.from_mat(cnv_u1).u.persist()

    # %%    
    #cnv_expr_cor = cnv_u1 @ (expr[['data']].assign(rows=lambda x: ('rows', np.random.permutation(x.rows.data))).data)
    cnv_expr_cor = cnv_u1 @ expr.data
    cnv_expr_cor = (cnv_expr_cor**2).rename('r2').to_dataframe().reset_index()
    print(cnv_expr_cor.groupby('cols').r2.sum().mean())
    sns.histplot(
        x='r2',
        data=cnv_expr_cor.groupby('cols').r2.sum().to_frame()
    )

    # %%
    #cnv_crispr_cor = cnv_u1 @ (crispr[['data']].assign(rows=lambda x: ('rows', np.random.permutation(x.rows.data))).data)
    cnv_crispr_cor = cnv_u1 @ crispr.data
    cnv_crispr_cor = (cnv_crispr_cor**2).rename('r2').to_dataframe().reset_index()
    print(cnv_crispr_cor.groupby('cols').r2.sum().mean())
    sns.histplot(
        x='r2',
        data=cnv_crispr_cor.groupby('cols').r2.sum().to_frame()
    )

# %%
if __name__ == '__main__':
    pass

    # %%
    playground11 = _playground11('full', 1)
    self = playground11

    # %%    