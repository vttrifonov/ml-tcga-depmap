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
data = self.data2

# %%
x1 = self.src
x1 = data[['src', x1, 'crispr']].rename({
    x1: 'data', 
    f'{x1}_cols': 'cols',
    'crispr': 'y',
    'crispr_rows': 'y_rows',
    'crispr_cols': 'y_cols'
}).persist()
x1['train'] = self._train_split.train
x1['y'] = x1.y.astype('float32')
x1['src_train'] = x1.src.to_series()+':'+x1.train.to_series().astype(str)

# %%
x2 = x1[['data', 'train']].sel(rows=x1.src=='dm')
x2 = x2.sel(rows=x2.train)
x2['y'] = x1.y.rename(y_rows='rows')
x2 = [
    scale(x).\
        rename('x').to_dataset().reset_coords(['center', 'scale']) 
    for x in [x2.data, x2.y]
]
x2 = [
    xa.merge([
        x.center, x.scale,
        SVD.from_mat(x.x).xarray
    ])
    for x in x2
]
x2 = [x.sel(pc=np.cumsum(x.s**2)/np.sum(x.s**2)<0.9) for x in x2]
x2 = [x.rename(pc=cols).persist() for x, cols in zip(x2, ['cols1', 'y_cols1'])]

# %%
x3 = [
    ((((d-x.center)/x.scale) @ x.v)/x.s).persist()
    for x, d in zip(x2, [x1.data, x1.y])
]

# %%
x4 = (x3[0]**2).sum(dim='cols1').rename('log_prob')
x4['src_train'] = x1.src_train

x5 = x2[0].u @ x2[1].u
x6 = x3[0] @ x5
x7 = xa.merge([
    x6.rename('pred').rename(rows='y_rows'), 
    x3[1].rename('obs'),
    x1.train.rename(rows='y_rows')
], join='inner').persist()

x8 = x7.groupby('train').apply(lambda x: xa.merge([
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


