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

expr_cyto = gdf.SVD.from_mat(m.expr.cyto_dummy).persist()

cnv_cyto = gdf.SVD.from_mat(m.cnv.cyto_dummy).persist()

x1 = m.cnv.data.T
x2 = cnv_cyto
m.cnv['cyto_coef'] = x2.inv(0).rmult(x1).usv.T.persist()
m.cnv['cyto_predict'] = ((x1.T @ x2.u) @ x2.u.T).persist()
m.cnv['cyto_resid'] = (x1.T-m.cnv.cyto_predict).persist()

d = m.cnv.data.assign_coords(arm=m.cnv.arm)
d = d.groupby('arm')
d = d.map(lambda x: (
    print(x.arm[0].values),
    xa.DataArray(
        daa.apply_gufunc(smooth, '(i),()->(i)', x, 40, vectorize=True),
        dims=x.dims,
        coords=x.coords
    )
)[1])
d = d.persist()
m.cnv['smooth'] = d.drop('arm')
m.cnv['smooth_resid'] = (m.cnv.data - m.cnv.smooth).persist()

px.scatter(
    (m.cnv.smooth_resid**2).mean(axis=0).rename('mean').\
        assign_coords(arm=m.cnv.arm, cyto=m.cnv.cyto).\
        to_dataframe().reset_index().reset_index().\
        query('mean>0.1')
    ,
    x='index', y='mean', color='arm',
    hover_data=['cyto', 'cols']
).show()

px.scatter(
    (m.cnv.smooth[0,:]).\
        assign_coords(data=m.cnv.data[0,:], arm=m.cnv.arm, cyto=m.cnv.cyto).\
        to_dataframe().reset_index().reset_index(),
    x='index', y=['smooth', 'data'], color='cyto',
    hover_data=['cyto', 'cols']
).show()


xa.merge([
    m.cnv.data.where(m.cnv.symbol.isin(['BRCA1', 'BRCA2', 'PARP1', 'PARP2']), drop=True).rename('expr'),
    m.crispr.data.where(m.crispr.cols.isin(['BRCA1', 'BRCA2', 'PARP1', 'PARP2']), drop=True).rename('crispr')
])

x2 = (x1 - x2).persist()
print((x1 ** 2).mean().values)
print((x2 ** 2).mean().values)

x1 = (m.cnv.smooth_resid**2).mean(axis=0).rename('mean').\
    assign_coords(arm=m.cnv.arm).to_dataframe().reset_index()
x1 = x1.query('mean>0.1')
x2 = x1.arm.to_numpy()
x2 = np.hstack([True, x2[1:]!=x2[:-1]])
x3 = x1.index.to_numpy()
x3 = np.hstack([True, x3[1:]!=(x3[:-1]+1)])
x1['f'] = np.cumsum(x2 | x3)
x1 = x1.sort_values(['f', 'mean'], ascending=[True, False])
x1 = x1.groupby('f').first()

resid = m.cnv.smooth_resid
#resid = resid.sel(cols=(m.cnv.smooth_resid**2).mean(axis=0)>0.1)
resid = resid.sel(cols=x1.cols.to_numpy())
resid.data = dmlp.StandardScaler().fit_transform(resid).data
resid.shape

x1 = m.crispr.data
x3 = gdf.concat1([
    resid
    #gdf.SVD.from_mat(m.expr.data).u[:,:400].persist().rename({'pc': 'cols'}),
    #gdf.SVD.from_mat(resid).u[:,:400].persist().rename({'pc': 'cols'})
])
x3 = gdf.SVD.from_mat(x3).persist()
x2 = x3.u[:,:400]
x2 = (x1 - x2 @ (x2.T @ x1)).persist()
#x4 = x3.perm.u[:,:400]
#x4 = (x1 - x4 @ (x4.T @ x1)).persist()
print((x2 ** 2).mean().values)
#print((x4 ** 2).mean().values)


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



