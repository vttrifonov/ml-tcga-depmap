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

import depmap_gdc_fit
importlib.reload(depmap_gdc_fit)
import depmap_gdc_fit as gdf

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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

def _svd1(d, cutoff, cols):
    x3 = d.fft_resid
    x3 = x3.assign_coords(fft_group=d.fft_group)
    x3 = x3.sel({cols: ~x3.fft_group.isnull()})
    x4 = d.fft.sel(freq=d.fft.fft_freq<cutoff)
    x4 = x4.rename({'freq': 'cols'}).drop(['fft_arm', 'fft_freq'])
    x4['fft_group'] = ('cols', np.zeros(x4.shape[1], dtype=int))
    x3 = xa.concat([x3, x4], 'cols')
    x3 = dmlp.StandardScaler().fit_transform(x3)
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

m = gdf.merge()._merge

d = m.dm_cnv.copy()
d['txMid'] = (d.txStart+d.txEnd)/2
d = d.sortby(['chrom', 'txMid'])
m.dm_cnv = d

d = m.gdc_cnv.copy()
d['txMid'] = (d.txStart+d.txEnd)/2
d.data.data = d.data.data.rechunk((None, -1))
with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    d = d.sortby(['chrom', 'txMid'])
m.gdc_cnv = d

d = _fft1(m.dm_cnv.data.assign_coords(arm=m.dm_cnv.arm), 10, 'arm', 'cols')
d = d.drop('arm')
m.dm_cnv = m.dm_cnv.merge(d)

d = _fft1(m.gdc_cnv.data.assign_coords(arm=m.gdc_cnv.arm), 10, 'arm', 'cols')
d = d.drop('arm')
m.gdc_cnv = m.gdc_cnv.merge(d)

x1 = (m.dm_cnv.fft_resid**2).mean(axis=0)
x2 = (m.gdc_cnv.fft_resid**2).mean(axis=0)
x3 = xa.merge([x1.rename('dm'), x2.rename('gdc')])
x3['arm'] = m.dm_cnv.arm
x3 = x3.to_dataframe().reset_index()
x3 = x3.query('dm>0.1 & gdc>0.1')
x3['g'] = _fft2(x3, 'arm')
m.dm_cnv['fft_group'] = x3.set_index('cols').g

np.corrcoef(x3.dm, x3.gdc)
x3['g1'] = x3.g.astype(str)
px.scatter(
    x3,
    x='dm', y='gdc', color='g1',
    hover_data=['cols']
).show()

m.dm_cnv_fft_svd = _svd1(m.dm_cnv, 10, 'cols')

#m.gdc_cnv_fft_svd = _svd1(m.gdc_cnv, 10, 'cols')

#expr_cyto = gdf.SVD.from_mat(m.expr.cyto_dummy).persist()

#x1 = m.dm_cnv.data.T
#x2 = gdf.SVD.from_mat(m.dm_cnv.cyto_dummy).persist()
#m.dm_cnv['cyto_coef'] = x2.inv(0).rmult(x1).usv.T.persist()
#m.dm_cnv['cyto_predict'] = ((x1.T @ x2.u) @ x2.u.T).persist()
#m.dm_cnv['cyto_resid'] = (x1.T-m.dm_cnv.cyto_predict).persist()

#d = _smooth1(m.dm_cnv.data.assign_coords(arm=m.dm_cnv.arm), 0.1, 'arm', 'cols')
#d = d.drop('arm')
#m.dm_cnv = m.dm_cnv.merge(d)

x1 = m.gdc_cnv
px.scatter(
    (x1.fft_resid**2).mean(axis=0).rename('mean').\
        assign_coords(arm=x1.arm, cyto=x1.cyto).\
        to_dataframe().reset_index().reset_index().\
        query('mean>=0')
    ,
    x='index', y='mean', color='arm',
    hover_data=['cyto', 'cols']
).show()

x1 = m.gdc_cnv
px.scatter(
    x1.rfft[0,:].\
        assign_coords(
            data=x1.data[0,:],
            arm=x1.arm, cyto=x1.cyto
        ).\
        to_dataframe().reset_index().reset_index(),
    x='index', y=['rfft', 'data'], color='arm',
    hover_data=['cyto', 'cols']
).show()


x1 = m.crispr.data.sel(rows=m.dm_cnv.rows.values).data.rechunk((-1, 1000)).to_zarr('tmp/crispr/mat')
x1 = daa.from_zarr('tmp/crispr/mat')
x2 = [x.u[:,:].data for x in m.dm_cnv_fft_svd.values()]
x2 = [((x1 - x @ (x.T @ x1))**2).mean(axis=0) for x in x2]
x2 = daa.stack(x2).persist()
x2 = x2.rechunk((-1, None))
x4 = [x.perm.u[:,:].data for x in m.dm_cnv_fft_svd.values()]
x4 = [((x1 - x @ (x.T @ x1))**2).mean(axis=0) for x in x4]
x4 = daa.stack(x4).persist()
x4 = x4.rechunk((-1, None))
x2.to_zarr('tmp/crispr/obs')
x4.to_zarr('tmp/crispr/rand')

x2 = daa.from_zarr('tmp/crispr/obs')
x4 = daa.from_zarr('tmp/crispr/rand')

x5 = dmlp.StandardScaler().fit(daa.log10(1-x4+1e-5).T)
x6 = x5.transform(daa.log10(1-x2+1e-5).T).T.persist()
m.crispr['cnv_rand_mean'] = ('fft_group', x5.mean_)
m.crispr['cnv_rand_var'] = ('fft_group', x5.var_)
m.crispr['cnv_r2'] = (('fft_group', 'cols'), x6)
m.crispr['fft_group'] = ('fft_group', np.arange(x2.shape[0]))

x5 = _fit(m.crispr.data, m.dm_expr.data, 400)
m.crispr = m.crispr.merge(x5.rename({'rand_mean': 'expr_rand_mean', 'rand_var': 'expr_rand_var', 'r2': 'expr_r2'}))

plt.figure()
plt.gca().plot(m.crispr.cnv_r2[0,:], m.crispr.expr_r2, '.')

plt.figure()
plt.gca().plot(m.crispr.cnv_r2[0,:], m.crispr.cnv_r2[1:,:].max(axis=0), '.')

plt.figure()
plt.gca().plot(m.crispr.expr_r2, m.crispr.cnv_r2[1:,:].mean(axis=0), '.')

plt.figure()
plt.gca().plot(m.crispr.cnv_r2[0,:], m.crispr.cnv_r2[80,:], '.')


pd.DataFrame(dict(
    expr=(m.crispr.expr_r2 > 2).values,
    cnv_glob = (m.crispr.cnv_r2[0,:]>2).values,
    cnv_loc = ((m.crispr.cnv_r2[1:,:]>2).sum(axis=0)>2).values
)).value_counts().sort_index()#.reset_index().pivot(index='loc', columns='glob')



x1 = m.crispr.data
x1 = gdf.SVD.from_mat(x1).persist()
x2 = m.expr.data
x2 = gdf.SVD.from_mat(x2).persist()
x3 = x2.cut(np.s_[:795]).u.T.rename({'pc': 'pc1'}) @ x1.cut(np.s_[:795]).u
x3 = x2.cut(np.s_[:794]).inv(0).vs.T.rename({'pc': 'pc1'}) @ x1.cut(np.s_[:794]).us.rename({'pc': 'pc2'})
x3 = gdf.SVD.from_mat(x3).persist()
x3 = x3.cut(np.s_[:400])
x3.u = x2.v.rename({'pc': 'pc1'}) @ x3.u
x3.v = x1.v.rename({'pc': 'pc2'}) @ x3.v
x3 = x3.persist()
x3 = x3.lmult(x2.usv).usv.persist()
x3 = (x1.usv - x3).persist()
print((x3 ** 2).mean().values)

plt.plot(x3.s)



