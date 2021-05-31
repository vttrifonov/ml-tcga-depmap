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
    if n > 0:
        rfft = daa.hstack([x[:, :n], daa.zeros((x.shape[0], x.shape[1] - n))])
    rfft = rfft.rechunk((None, -1))
    rfft = daa.fft.irfft(rfft)
    rfft = rfft[:, :x.shape[1]]
    return rfft

def _data():
    m = gdf.merge()

    crispr = m.crispr1.copy()
    cnv = m.dm_cnv1.copy()
    expr = m.dm_expr1.copy()
    rows = set(crispr.rows.values) & set(cnv.rows.values) & set(expr.rows.values)
    rows = list(rows)
    crispr = crispr.sel(rows=crispr.rows.isin(rows))
    cnv = cnv.sel(rows=cnv.rows.isin(rows))
    expr = expr.sel(rows=expr.rows.isin(rows))

    crispr['data'] = dmlp.StandardScaler().fit_transform(crispr.data.astype('float32'))
    cnv['data'] = dmlp.StandardScaler().fit_transform(cnv.data.astype('float32'))
    expr['data'] = dmlp.StandardScaler().fit_transform(expr.data.astype('float32'))

    return SimpleNamespace(
        crispr = m.crispr,
        cnv = cnv,
        expr = expr
    )

m = gdf.merge()._merge
d = m.dm_cnv
d = d.sel(cols=~d['loc'].isnull())
d = d.sel(cols=~d.arm.isin(['--', 'X;YX;Y']))
d = d.sortby(['chr', 'loc'])
m.dm_cnv = d

def _cnv_fft():
    d = m.dm_cnv.data.assign_coords(arm=m.dm_cnv.arm).groupby('arm')
    d = [
        xa.apply_ufunc(
            _fft, x,
            input_core_dims = [['cols']], output_core_dims=[['freq']],
            dask='allowed'
        ).assign_coords({
            'fft_arm': ('freq', x.arm),
            'fft_freq': ('freq', np.arange(x.shape[1]))
        }).assign_coords({
            'freq': lambda x: ('freq', x.fft_arm + ':' + x.fft_freq.astype(str))
        })
        for arm, x in d
    ]
    d = xa.concat(d, 'freq')
    d.data = d.data.rechunk(m.dm_cnv.data.data.chunks)
    m.dm_cnv['fft'] = d.persist()
    print('hi1')

    d = m.dm_cnv.fft.groupby('fft_arm')
    d = [
        xa.apply_ufunc(
            _rfft, x, min(10, x.shape[1]),
            input_core_dims = [['freq'], []], output_core_dims=[['cols']],
            dask='allowed'
        ).assign_coords({
            'cols': m.dm_cnv.cols.sel(cols=m.dm_cnv.arm==arm)
        })
        for arm, x in d
    ]
    d = xa.concat(d, 'cols')
    m.dm_cnv['rfft'] = d.persist()
    m.dm_cnv['fft_resid'] = (m.dm_cnv.data - m.dm_cnv['rfft']).persist()
    print('hi2')

    x1 = (m.dm_cnv.fft_resid**2).mean(axis=0).rename('mean').\
        assign_coords(arm=m.dm_cnv.arm).to_dataframe().reset_index()
    x1 = x1.query('mean>0.1').copy()
    x2 = x1.arm.to_numpy()
    x2 = np.hstack([True, x2[1:]!=x2[:-1]])
    x3 = x1.index.to_numpy()
    x3 = np.hstack([True, x3[1:]!=(x3[:-1]+1)])
    x1['f'] = np.cumsum(x2 | x3)
    x4 = np.zeros(m.dm_cnv.cols.shape[0])
    x4[x1.index] = x1.f
    m.dm_cnv['fft_group'] = ('cols', x4)
    print('hi3')

    x3 = m.dm_cnv.fft_resid
    x3 = x3.assign_coords(fft_group=m.dm_cnv.fft_group)
    x3 = x3.sel(cols=x3.fft_group>0)
    x4 = m.dm_cnv.fft.sel(freq=m.dm_cnv.fft.fft_freq<10)
    x4 = x4.rename({'freq': 'cols'}).drop(['fft_arm', 'fft_freq'])
    x4['fft_group'] = ('cols', np.zeros(x4.shape[1]))
    x3 = xa.concat([x3, x4], 'cols')
    x3 = dmlp.StandardScaler().fit_transform(x3)
    x3 = x3.groupby('fft_group')
    x3 = {k: gdf.SVD.from_mat(x).persist() for k, x in x3}
    m.dm_cnv_fft_svd = x3
    print('hi4')

_cnv_fft()

#expr_cyto = gdf.SVD.from_mat(m.expr.cyto_dummy).persist()

cnv_cyto = gdf.SVD.from_mat(m.dm_cnv.cyto_dummy).persist()

x1 = m.dm_cnv.data.T
x2 = cnv_cyto
m.dm_cnv['cyto_coef'] = x2.inv(0).rmult(x1).usv.T.persist()
m.dm_cnv['cyto_predict'] = ((x1.T @ x2.u) @ x2.u.T).persist()
m.dm_cnv['cyto_resid'] = (x1.T-m.dm_cnv.cyto_predict).persist()

d = m.dm_cnv.data.assign_coords(arm=m.dm_cnv.arm)
d = d.groupby('arm')
d = d.map(lambda x: (
    xa.apply_ufunc(
        smooth, x, int(0.1*x.shape[1]),
        input_core_dims = [['cols'], []], output_core_dims=[['cols']],
        dask='parallelized', vectorize=True
    )
))
d = d.persist().drop('arm')
d.data = d.data.rechunk((None, -1))
m.dm_cnv['smooth'] = d
m.dm_cnv['smooth_resid'] = (m.dm_cnv.data - m.dm_cnv.smooth).persist()

px.scatter(
    (m.dm_cnv.smooth_resid**2).mean(axis=0).rename('mean').\
        assign_coords(arm=m.dm_cnv.arm, cyto=m.dm_cnv.cyto).\
        to_dataframe().reset_index().reset_index().\
        query('mean>=0')
    ,
    x='index', y='mean', color='arm',
    hover_data=['cyto', 'cols']
).show()

px.scatter(
    m.dm_cnv.rfft[0,:].\
        assign_coords(
            data=m.dm_cnv.data[0,:],
            arm=m.dm_cnv.arm, cyto=m.dm_cnv.cyto
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

x5 = x4.min(axis=1).reshape(-1,1)
x6 = (x2<x5).persist()
plt.plot(x6.sum(axis=0), '.')

pd.DataFrame(dict(n=x6.sum(axis=1).compute())).query('n>10')

pd.DataFrame(dict(
    cols=m.crispr.cols.values,
    n=x6.sum(axis=0).compute()
)).query('n>1').sort_values('n').shape

#816
m.cnv.cyto.sel(cols=m.cnv.fft_group==816)

plt.plot(
    sorted(x2[0,:].compute()),
    sorted(x4[0,:].compute()),
    '.'
)
plt.gca().axline((1,1), slope=1)


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



