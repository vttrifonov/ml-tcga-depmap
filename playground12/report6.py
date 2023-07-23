# %%
import sys
from pathlib import Path
path = str(Path.cwd().parent.parent)
if not path in sys.path:
    sys.path = [path] + sys.path
__package__ = 'ml_tcga_depmap.playground12'

import plotnine as p9
import xarray as xa
import numpy as np
import pandas as pd
import dask as da
import matplotlib.pyplot as plt

from ..common.caching import compose, lazy
from ..svd import SVD
from . import _analysis, _scale1 as scale

# %%
class _analysis6(_analysis):    
    def __init__(self, *args, src):
        super().__init__(*args)
        self.src = src

# %%
analysis6 = _analysis6('20230531/0.5', 0.5, src='expr')
self = analysis6

# %%
src = self.src
data = self.data1
data = data[['src', src, 'crispr']].rename({
    src: 'x', 
    'rows': 'x_rows',
    f'{src}_cols': 'x_cols',    
    'crispr': 'y',
    'crispr_rows': 'y_rows',
    'crispr_cols': 'y_cols'
}).persist()

# %%
x1 = data.copy()
x1['train'] = self._train_split.train.rename(rows='x_rows')
x1['src_train'] = x1.src.to_series()+':'+x1.train.to_series().astype(str)

# %%
x2 = x1.x.sel(x_rows=x1.train)
x2 = scale(x2, rows=['x_rows']).rename('data').reset_coords(['center', 'scale']).persist()
x3 = SVD.from_mat(x2.data, n=1000, solver='rand').xarray.persist()
x3 = x3.sel(pc=(np.cumsum(x3.s**2)/np.sum(x2.data**2)<0.9).compute())
x1 = xa.merge([
    x1.drop_dims('x_cols'),
    ((((x1.x-x2.center)/x2.scale)@x3.v)).rename('x', pc='x_cols')
]).persist()

# %%
x2 = x1.train.sel(x_rows=x1.y_rows).drop('x_rows')
x2 = x1.y.sel(y_rows=x2)
x2 = scale(x2, rows=['y_rows']).rename('data').reset_coords(['center', 'scale']).persist()
x3 = SVD.from_mat(x2.data, n=400, solver='rand').xarray.persist()
x3 = x3.sel(pc=(np.cumsum(x3.s**2)/np.sum(x2.data**2)<0.9).compute())
x1 = xa.merge([
    x1.drop_dims('y_cols'),
    ((((x1.y-x2.center)/x2.scale)@x3.v)).rename('y', pc='y_cols')
]).persist()

# %%
x2 = x1[['x', 'y']].sel(x_rows=x1.src_train=='dm:True')
x2 = x2.sel(y_rows=x2.x_rows).drop('y_rows')
def _(x, cols):
    m = x.mean(dim='x_rows')
    x = x-m
    s = np.sqrt((x**2).mean(dim='x_rows'))
    x = x/s
    x = SVD.from_mat(x).xarray
    #x = x.sel(pc=np.cumsum(x.s**2)/np.sum(x.s**2)<0.9)
    x = x.rename(pc=cols).persist()
    return xa.merge([x, m.rename('center'), s.rename('scale')])
x2 = [_(*x).persist() for x in zip([x2.x, x2.y], ['x_cols1', 'y_cols1'])]

# %%
x, d = x2[0], x1.x
z1 = ((((d-x.center)/x.scale)@x.v)/x.s).compute()
z1.groupby(x1.src_train).apply(lambda x: (x**2).sum(dim='x_cols1').mean(dim='x_rows'))

# %%
x3 = [
    ((((d-x.center)/x.scale) @ x.v)/x.s).persist()
    for x, d in zip(x2, [x1.x, x1.y])
]

# %%
x4 = (x3[0]**2).sum(dim='x_cols1').rename('log_prob')
x4['src_train'] = x1.src_train

x5 = x2[0].u @ x2[1].u

x6 = x3[0] @ x5
x6 = xa.merge([
    x6.rename('pred').rename(x_rows='y_rows'), 
    x3[1].rename('obs'),
    x1.train.rename(x_rows='y_rows')
], join='inner').persist()

x8 = x6.groupby('train').apply(lambda x: xa.merge([
    ((x.obs-x.pred)**2).sum(dim='y_rows').rename('delta'),
    (x.obs**2).sum(dim='y_rows').rename('obs')
])).persist()

# %%
x9 = x4.to_dataset().to_dataframe()
(
    p9.ggplot(x9)+
        p9.aes('src_train', 'log_prob')+
        p9.geom_boxplot()
)

# %%
x9 = x8.to_dataframe().reset_index()
(
    p9.ggplot(x9)+
        p9.aes('obs', 'delta', color='train')+
        p9.geom_point()
)


