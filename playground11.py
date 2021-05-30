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

    cnv['data'] = daa.log2(cnv.data.astype('float32')+0.1)
    crispr['data'] = dmlp.StandardScaler().fit_transform(crispr.data.astype('float32'))
    cnv['data'] = dmlp.StandardScaler().fit_transform(cnv.data.astype('float32'))
    expr['data'] = dmlp.StandardScaler().fit_transform(expr.data.astype('float32'))

    def _add_locs(x):
        x = x.rename({'map_location': 'cyto'})
        map = x.cyto.to_dataframe()
        map['chr'] = map.cyto.str.replace('[pq].*$', '', regex=True)
        map['pq'] = map.cyto.str.replace('^.*([pq]).*$', r'\1', regex=True)
        map['loc'] = map.cyto.str.replace('^.*[pq]', '', regex=True)
        map['loc'] = pd.to_numeric(map['loc'], errors='coerce')
        map['loc'] = (2*(map.pq=='q')-1)*map['loc']
        map['arm'] = map.chr + map.pq
        map = map.sort_values(['chr', 'loc'])
        x = x.merge(map[['chr', 'loc', 'arm']])
        x = x.sel(cols=map.index)
        x = x.sel(cols=~x['loc'].isnull())
        x.data.values = x.data.data.rechunk('auto').persist()
        return x

    cnv = _add_locs(cnv)
    expr = _add_locs(expr)

    def _loc_dummy(x):
        import sparse
        import re

        loc = x.to_series()
        loc = loc.str.replace(' and ', '|')
        loc1 = [np.array(x.split('|')) for x in loc]
        loc2 = [np.repeat(loc.index[i], len(x)) for i, x in enumerate(loc1)]
        loc = pd.Series(
            np.hstack(loc1),
            index=pd.Index(np.hstack(loc2), name=loc.index.name),
            name=loc.name
        )
        loc = loc.reset_index().drop_duplicates().set_index(loc.index.name).squeeze()

        def _split(s):
            s = s.split('-')
            if len(s)==1:
                return s
            chr = re.sub('[pq].*$', '', s[0])
            s[1] = chr + s[1]
            return s

        loc1 = [np.array(_split(x)) for x in loc]
        loc2 = [np.repeat(loc.index[i], len(x)) for i, x in enumerate(loc1)]
        loc = pd.Series(
            np.hstack(loc1),
            index=pd.Index(np.hstack(loc2), name=loc.index.name),
            name=loc.name
        )
        loc = loc.reset_index().drop_duplicates().set_index(loc.index.name).squeeze()



        rows = pd.Series(range(len(x.cols)), index=x.cols)
        cols = loc.drop_duplicates()
        cols = pd.Series(range(len(cols)), index=cols)
        data = sparse.COO([rows[loc.index], cols[loc]], 1, shape=(len(rows), len(cols)))
        data = data.todense()
        data = daa.from_array(data)
        loc = xa.DataArray(
            data,
            dims=('cols', x.name + '_cols'),
            coords={
                'cols': rows.index.to_numpy(),
                x.name + '_cols': cols.index.to_numpy()
            },
            name=x.name
        )
        return loc

    #cnv['arm_dummy'] = _loc_dummy(cnv.arm)
    cnv['cyto_dummy'] = _loc_dummy(cnv.cyto)
    #expr['arm_dummy'] = _loc_dummy(expr.arm)
    expr['cyto_dummy'] = _loc_dummy(expr.cyto)

    return SimpleNamespace(
        crispr = crispr,
        cnv = cnv,
        expr = expr
    )

m = _data()

def _cnv_fft():
    d = m.cnv.data.assign_coords(arm=m.cnv.arm).groupby('arm')
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
    d.data = d.data.rechunk(m.cnv.data.data.chunks)
    m.cnv['fft'] = d.persist()
    print('hi1')

    d = m.cnv.fft.groupby('fft_arm')
    d = [
        xa.apply_ufunc(
            _rfft, x, min(10, x.shape[1]),
            input_core_dims = [['freq'], []], output_core_dims=[['cols']],
            dask='allowed'
        ).assign_coords({
            'cols': m.cnv.cols.sel(cols=m.cnv.arm==arm)
        })
        for arm, x in d
    ]
    d = xa.concat(d, 'cols')
    m.cnv['rfft'] = d.persist()
    m.cnv['fft_resid'] = (m.cnv.data - m.cnv['rfft']).persist()
    print('hi2')

    x1 = (m.cnv.fft_resid**2).mean(axis=0).rename('mean').\
        assign_coords(arm=m.cnv.arm).to_dataframe().reset_index()
    x1 = x1.query('mean>0.1').copy()
    x2 = x1.arm.to_numpy()
    x2 = np.hstack([True, x2[1:]!=x2[:-1]])
    x3 = x1.index.to_numpy()
    x3 = np.hstack([True, x3[1:]!=(x3[:-1]+1)])
    x1['f'] = np.cumsum(x2 | x3)
    x4 = np.zeros(m.cnv.cols.shape[0])
    x4[x1.index] = x1.f
    m.cnv['fft_group'] = ('cols', x4)
    print('hi3')

    x3 = m.cnv.fft_resid
    x3 = x3.assign_coords(fft_group=m.cnv.fft_group)
    x3 = x3.sel(cols=x3.fft_group>0)
    x4 = m.cnv.fft.sel(freq=m.cnv.fft.fft_freq<10)
    x4 = x4.rename({'freq': 'cols'}).drop(['fft_arm', 'fft_freq'])
    x4['fft_group'] = ('cols', np.zeros(x4.shape[1]))
    x3 = xa.concat([x3, x4], 'cols')
    x3 = dmlp.StandardScaler().fit_transform(x3)
    x3 = x3.groupby('fft_group')
    x3 = {k: gdf.SVD.from_mat(x).persist() for k, x in x3}
    m.cnv_fft_svd = x3
    print('hi4')

_cnv_fft()

#expr_cyto = gdf.SVD.from_mat(m.expr.cyto_dummy).persist()

cnv_cyto = gdf.SVD.from_mat(m.cnv.cyto_dummy).persist()

x1 = m.cnv.data.T
x2 = cnv_cyto
m.cnv['cyto_coef'] = x2.inv(0).rmult(x1).usv.T.persist()
m.cnv['cyto_predict'] = ((x1.T @ x2.u) @ x2.u.T).persist()
m.cnv['cyto_resid'] = (x1.T-m.cnv.cyto_predict).persist()

d = m.cnv.data.assign_coords(arm=m.cnv.arm).groupby('arm')
d = d.map(lambda x: (
    xa.apply_ufunc(
        smooth, x, int(0.1*x.shape[1]),
        input_core_dims = [['cols'], []], output_core_dims=[['cols']],
        dask='parallelized', vectorize=True
    )
))
m.cnv['smooth'] = d.persist().drop('arm')
m.cnv['smooth_resid'] = (m.cnv.data - m.cnv.smooth).persist()

px.scatter(
    (m.cnv.smooth_resid**2).mean(axis=0).rename('mean').\
        assign_coords(arm=m.cnv.arm, cyto=m.cnv.cyto).\
        to_dataframe().reset_index().reset_index().\
        query('mean>=0')
    ,
    x='index', y='mean', color='arm',
    hover_data=['cyto', 'cols']
).show()

px.scatter(
    (m.cnv.rfft[0,:]).\
        assign_coords(data=m.cnv.data[0,:], arm=m.cnv.arm, cyto=m.cnv.cyto).\
        to_dataframe().reset_index().reset_index(),
    x='index', y=['rfft', 'data'], color='arm',
    hover_data=['cyto', 'cols']
).show()

x1 = m.crispr.data.sel(rows=m.cnv.rows.values).data.rechunk((-1, 1000)).to_zarr('tmp/crispr/mat')
x1 = daa.from_zarr('tmp/crispr/mat')
x2 = [x.u[:,:].data for x in m.cnv_fft_svd.values()]
x2 = [((x1 - x @ (x.T @ x1))**2).mean(axis=0) for x in x2]
x2 = daa.stack(x2).persist()
x2 = x2.rechunk((-1, None))
x4 = [x.perm.u[:,:].data for x in m.cnv_fft_svd.values()]
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



