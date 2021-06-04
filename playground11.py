import depmap_gdc_fit as gdf
import matplotlib.pyplot as plt
import seaborn as sns
from common.dir import cached_property, Dir
from common.defs import lazy_property
from types import SimpleNamespace
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

import depmap_gdc_fit
importlib.reload(depmap_gdc_fit)
import depmap_gdc_fit as gdf

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

config.exec()

def smooth(x, window_len=11, window='hanning'):
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
    x4['fft_group'] = ('cols', np.zeros(x4.shape[1], dtype=int))
    x3 = xa.concat([x3, x4], 'cols')
    x3 = dmlp.StandardScaler().fit_transform(x3)
    if rand:
        x3.data = daa.apply_along_axis(np.random.permutation, 0, x3.data, shape=(x3.shape[0],), dtype=x3.dtype)
    x3 = x3.groupby('fft_group')
    x3 = {k: gdf.SVD.from_mat(x) for k, x in x3}
    return x3

def _fit(x1, x3, cutoff):
    x3 = gdf.SVD.from_mat(x3).persist()
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
            smooth, x, int(frac*x.shape[1]),
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

class _playground11:
    @lazy_property
    def m(self):
        return gdf.merge()._merge
playground11 = _playground11()

def _():
    _playground11.storage = config.cache/'playground11'
    _playground11.crispr = property(lambda self: self.m.crispr)
    _playground11.dm_expr = property(lambda self: self.m.dm_expr)

    def _dm_cnv(self):
        m = self.m
        d = m.dm_cnv.copy()
        d['txMid'] = (d.txStart+d.txEnd)/2
        d = d.sortby(['chrom', 'txMid'])
        return d
    _playground11.dm_cnv = lazy_property(_dm_cnv)
    #del self.__lazy___dm_cnv

    def _gdc_cnv(self):
        m = self.m
        d = m.gdc_cnv.copy()
        d['txMid'] = (d.txStart + d.txEnd) / 2
        d.data.data = d.data.data.rechunk((None, -1))
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            d = d.sortby(['chrom', 'txMid'])
        return d
    _playground11.gdc_cnv = lazy_property(_gdc_cnv)
    #del self.__lazy___gdc_cnv

    def _dm_cnv_fft(self):
        d = _fft1(self.dm_cnv.data.assign_coords(arm=self.dm_cnv.arm), 10, 'arm', 'cols')
        d = d.drop('arm')
        return d
    _playground11.dm_cnv_fft = lazy_property(_dm_cnv_fft)
    #del self.__lazy___dm_cnv_fft

    def _gdc_cnv_fft(self):
        d = _fft1(self.gdc_cnv.data.assign_coords(arm=self.gdc_cnv.arm), 10, 'arm', 'cols')
        d = d.drop('arm')
        return d
    _playground11.gdc_cnv_fft = lazy_property(_gdc_cnv_fft)
    #del self.__lazy___gdc_cnv_fft

    def _dm_cnv_fft_group(self):
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
    _playground11.dm_cnv_fft_group = lazy_property(_dm_cnv_fft_group)
    #del self.__lazy___dm_cnv_fft_group

    def _dm_cnv_fft_svd(self):
        fft = self.dm_cnv_fft
        fft['fft_group'] = self.dm_cnv_fft_group
        return _svd1(fft, 10, 'cols')
    _playground11.dm_cnv_fft_svd = lazy_property(_dm_cnv_fft_svd)
    # del self.__lazy___dm_cnv_fft_svd

    def _dm_cnv_fft_svd_rand(self):
        fft = self.dm_cnv_fft
        fft['fft_group'] = self.dm_cnv_fft_group
        return _svd1(fft, 10, 'cols', True)
    _playground11.dm_cnv_fft_svd_rand = lazy_property(_dm_cnv_fft_svd_rand)
    # del self.__lazy___dm_cnv_fft_svd_rand

    def _dm_cnv_fft_svd_pc(self):
        m = self
        x1 = [x.ve.data for x in m.dm_cnv_fft_svd.values()]
        x1 = daa.hstack(x1)
        x1 = x1.compute()
        x4 = [x.ve.data for x in m.dm_cnv_fft_svd_rand.values()]
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
    _playground11.dm_cnv_fft_svd_pc = lazy_property(_dm_cnv_fft_svd_pc)
    # del self.__lazy___dm_cnv_fft_svd_pc

    def _crispr_cnv_fit(self):
        s = self.storage/'crispr'/'cnv_fit'
        if not s.exists():
            x1 = self.crispr.data.sel(rows=self.dm_cnv.rows.values).data.rechunk((-1, 1000)).to_zarr(str(s/'mat'))
            x1 = daa.from_zarr(str(s/'mat'))
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
        #m.crispr = m.crispr.drop(['fft_group', 'cnv_rand_mean', 'cnv_rand_var', 'cnv_r2'])
        m = xa.Dataset()
        m['fft_group'] = ('fft_group', np.arange(x2.shape[0]))
        m['cols'] = self.crispr.cols
        m['rand_mean'] = ('fft_group', x5.mean_)
        m['rand_var'] = ('fft_group', x5.var_)
        m['r2'] = (('fft_group', 'cols'), x6)
        return m
    _playground11.crispr_cnv_fit = lazy_property(_crispr_cnv_fit)
    # del self.__lazy___crispr_cnv_fit

    def _crispr_expr_fit(self):
        return _fit(self.crispr.data, self.dm_expr.data, 400)
    _playground11.crispr_expr_fit = lazy_property(_crispr_expr_fit)
    # del self.__lazy___crispr_expr_fit
_()

plot_fft_resid(playground11.dm_cnv.merge(playground11.dm_cnv_fft))
plot_fft(playground11.dm_cnv.merge(playground11.dm_cnv_fft))

m = playground11
(m.crispr_cnv_fit.r2[1:,:]>3).sum(axis=1).to_series().sort_values()

m.crispr_cnv_fit.r2[28,:].to_series().pipe(lambda x: x[x>3]).sort_values()

m.crispr.cols.sel(cols=m.crispr.symbol=='TP53')

(m.crispr_cnv_fit.r2[1:,:]>2).sum(axis=0).to_series().sort_values().pipe(lambda x: x[x>0])['TP53 (7157)']


m = playground11
x5 = 'TP53 (7157)'
x1 = m.crispr_cnv_fit.cnv_r2.loc[1:,x5].to_series().pipe(lambda x: x[x>2]).index
x1 = [0] + list(x1)
x1 = [x.u[:,:int(m.dm_cnv_fft_svd_pc[int(i)].item())] for i, x in m.dm_cnv_fft_svd.items() if i in x1]
x6 = m.dm_expr.data
x6 = x6 - x1[0] @ (x1[0].T @ x6)
x6 = gdf.SVD.from_mat(x6).u[:,:300].persist()
x1 = x1 + [x6]
x1 = [x.assign_coords(pc=([str(i)+':'+x for x in x.pc.values.astype(str)])) for i, x in enumerate(x1)]
x1 = xa.concat(x1, 'pc').rename({'pc': 'cols'})
x1 = gdf.SVD.from_mat(x1).persist()
x2 = m.crispr.data.loc[:,x5].compute()
x3 = x1.u
x3 = (x3 @ (x3.T @ x2)).compute()
plt.plot(x2, x3, '.')
print((1-(x2-x3)**2).mean().values.round(2), np.array(x1.u.shape[1]/x1.u.shape[0]).round(2))


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



