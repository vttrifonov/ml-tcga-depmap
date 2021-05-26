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
        loc = x.to_dataframe()
        loc = pd.get_dummies(loc, prefix='', prefix_sep='')
        loc = xa.DataArray(
            loc,
            dims=('cols', x.name + '_cols'),
            coords={
                'cols': loc.index,
                x.name + '_cols': loc.columns
            },
            name=x.name
        )
        loc.values = daa.from_array(loc.values)
        return loc

    cnv['arm_dummy'] = _loc_dummy(cnv.arm)
    cnv['cyto_dummy'] = _loc_dummy(cnv.cyto)
    expr['arm_dummy'] = _loc_dummy(expr.arm)
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
x2 = x2.u[:,:].persist()
x2 = x2 @ (x2.T @ x1).persist()
x2 = (x1 - x2).persist()
print((x1 ** 2).mean().values)
print((x2 ** 2).mean().values)

resid = x2.T

x1 = m.crispr.data
x3 = gdf.concat1([
    gdf.SVD.from_mat(m.expr.data).u[:,:400].persist().rename({'pc': 'cols'}),
    gdf.SVD.from_mat(resid).u[:,:100].persist().rename({'pc': 'cols'})
])
x3 = gdf.SVD.from_mat(x3).persist()
x2 = x3.u[:,:400]
x2 = (x1 - x2 @ (x2.T @ x1)).persist()
#x4 = x3.perm.u[:,:400]
#x4 = (x1 - x4 @ (x4.T @ x1)).persist()
print((x1 ** 2).mean().values)
print((x2 ** 2).mean().values)
#print((x4 ** 2).mean().values)

px.scatter(
    pd.DataFrame(dict(
        obs=sorted((x2 ** 2).mean(axis=0).values),
        rand=sorted((x4 ** 2).mean(axis=0).values)
    )),
    x='obs', y='rand'
).show()

pd.DataFrame(dict(
    cols=m.crispr.cols.values,
    obs=(x2 ** 2).mean(axis=0).values,
    rand=(x4 ** 2).mean(axis=0).values
)).sort_values('obs')


xx = m.expr.data.loc['ACH-001850', :]. \
    assign_coords({'cyto': m.expr.cyto}). \
    assign_coords({'arm': m.expr.arm}). \
    assign_coords({'cnv': x2.loc['ACH-001850', :]}). \
    to_dataframe().reset_index().reset_index()

px.scatter(
    xx,
    x='cnv', y='data', color='arm',
    hover_data=['cyto']
).show()


