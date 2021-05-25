import depmap_gdc_fit as gdf
import matplotlib.pyplot as plt
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

import depmap_gdc_fit
importlib.reload(depmap_gdc_fit)
import depmap_gdc_fit as gdf

m = gdf.merge()

cnv = m.dm_cnv.mat.copy()
map = cnv.map_location.to_dataframe()
map['chr'] = map.map_location.str.replace('[pq].*$', '', regex=True)
map['pq'] = map.map_location.str.replace('^.*([pq]).*$', r'\1', regex=True)
map['loc'] = map.map_location.str.replace('^.*[pq]', '', regex=True)
map['loc'] = pd.to_numeric(map['loc'], errors='coerce')
map['loc'] = (2*(map.pq=='p')-1)*map['loc']
map = map.sort_values(['chr', 'loc'])
cnv = cnv.merge(map[['chr', 'loc']])
cnv = cnv.sel(cols=map.index)
cnv = cnv.sel(cols=~cnv['loc'].isnull())
cnv.data.values = cnv.data.data.rechunk('auto').persist()

cyto = cnv.map_location.to_dataframe()
cyto = pd.get_dummies(cyto)
cyto = xa.DataArray(
    cyto,
    dims=('cols', 'locs'),
    coords=dict(
        cols = cyto.index,
        locs = cyto.columns
    ),
    name='cytoband'
)
cyto.values = daa.from_array(cyto.values)
cyto = cyto.sel(locs=cyto.sum(axis=0)>10)
cyto = cyto.sel(cols=cyto.sum(axis=1)>0)
cyto.data = dmlp.StandardScaler().fit_transform(cyto.data)
cyto = SimpleNamespace(
    mat = cyto,
    svd = gdf.SVD.from_mat(cyto).persist()
)

cnv = cnv.sel(cols=cyto.mat.cols)

x1 = cnv.data.T
x2 = cyto.svd
x2 = x2.u @ (x2.u.T @ x1)
cnv['resid'] = (x1 - x2).T.persist()
(cnv.resid**2).mean().compute()
plt.plot(cnv.resid.mean(axis=0), cnv.resid.std(axis=0), '.')
cnv['resid'] = dmlp.StandardScaler().fit_transform(cnv.resid).persist()
resid = gdf.Mat(lambda: cnv[['resid']].rename({'resid': 'data'}))

x1 = m.crispr.mat.data
x2 = resid.svd.cut(np.s_[:500])
x2 = x2.u @ (x2.u.T @ x1)
x2 = x2.persist()

x3 = resid.svd.cut(np.s_[:500]).perm
x3 = x3.u @ (x3.u.T @ x1)
x3 = x3.persist()

plt.plot(
    sorted(((x1-x2)**2).mean(axis=0).values),
    sorted(((x1-x3)**2).mean(axis=0).values),
    '.'
)

