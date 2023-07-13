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
import dask
import matplotlib.pyplot as plt

from ..common.caching import compose, lazy
from ..svd import SVD
from . import _analysis, _scale2 as scale

# %%
analysis = _analysis('20230531/0.8', 0.8)

# %%
data = analysis.data2

# %%
x1 = data[['train', 'src', 'cnv']].rename(cnv='data', cnv_cols='cols')
x1 = x1.persist()
x1 = xa.merge([x1.drop('data'), scale(x1.data).rename('data')], join='inner')
x1 = x1.persist()
x1 = x1.transpose('rows', 'cols')

# %%
x2 = SVD.from_mat(x1.data, n=200, solver='rand').us.persist()
x2 = xa.merge([x2.rename('data'), x1[['src', 'train']]])
x2['src_train'] = x2.src.to_series()+':'+x2.train.to_series().astype(str)
x2 = x2.rename(pc='cols1')

# %%
x3 = x2.groupby('src_train')
n, x = next(iter(x3))
def _(n, x):
    x['m'] = x.data.mean(dim='rows')
    x['data'] = x.data-x.m
    x['data'] = x.data/np.sqrt(x.sizes['rows'])
    cov = x.data.rename(cols1='cols1_1') @ x.data
    cov = cov + np.diag([1e-6]*cov.shape[0])
    cov = SVD.from_mat(cov).xarray[['s', 'v']]
    cov['s'] = np.sqrt(cov.s)
    x = xa.merge([x.drop_dims('rows'), cov])
    x = x.expand_dims(src_train=[n])
    return x
x3 = xa.concat([_(n, x) for n, x in x3], dim='src_train')
x3 = x3.persist()

# %%
x4 = (x3.v*x3.s)
x5 = (x3.v/x3.s).rename(pc='pc1', src_train='src_train1')
x6 = x4 @ x5
x6 = xa.dot(x6, x6, dims=['pc', 'pc1'])

x7 = x3.m - x3.m.rename(src_train='src_train1')
x7 = xa.dot(x7, x5, dims='cols1')
x7 = xa.dot(x7, x7, dims='pc1')

x8 = np.log(x3.s).sum(dim='pc')
x8 = x8.rename(src_train='src_train1')-x8

x9 = 0.5*(x6 + x7 - x3.sizes['cols1'])+x8
x9 = x9.rename('kl').to_dataframe().reset_index()

# %%
(
    p9.ggplot(x9)+
        p9.aes(
            'src_train', 'src_train1', 
            fill='np.log10(kl+1)', label='kl.astype(int)'
        )+
        p9.geom_tile()+
        p9.geom_text()
)

# %%
x3 = x2.groupby('src_train')
n, x = next(iter(x3))
def _(n, x):
    x['m'] = x.data.mean(dim='rows')
    x['data'] = x.data-x.m
    x['data'] = x.data/np.sqrt(x.sizes['rows'])
    cov = x.data.transpose('rows', 'cols1')
    x = x.drop_dims('rows').expand_dims(src_train1=[n])
    cov = SVD.from_mat(cov).xarray[['s', 'v']]
    cov = cov.expand_dims(src_train=[n])
    cov = cov.stack(src_train_pc=['src_train', 'pc'], create_index=False)    
    return x, cov
x3 = [_(*x) for x in x3]
x3 = list(zip(*x3))
x3 = xa.merge([
    xa.concat(x, dim=d) 
    for x, d in zip(x3, ['src_train1', 'src_train_pc'])
])
x3 = x3.sel(src_train_pc=(x3.s>1e-3).compute())
x3 = x3.persist()

# %%
x6 = (x3.v*x3.s).drop('pc')
x5 = (x3.v/x3.s).drop('pc').rename(
    src_train_pc='src_train_pc1',
    src_train='src_train1'
)

x6 @= x5
x6 *= x6
x6 = x6.rename('x').to_dataframe()
x6 = x6.groupby(['src_train', 'src_train1']).x.sum()
x6 = x6.to_xarray()

x7 = x3.m.rename(src_train1='src_train') @ x5
x7 -= x7.where(x7.src_train==x7.src_train1, 0).sum(dim='src_train')
x7 *= x7
x7 = x7.rename('x').to_dataframe()
x7 = x7.groupby(['src_train', 'src_train1']).x.sum()
x7 = x7.to_xarray()

x8 = np.log(x3.s)
x8 = x8.rename('x').to_dataframe()
x8 = x8.groupby('src_train').x.sum().to_xarray()
x8 = x8.rename(src_train='src_train1')-x8

x10 = x3[['src_train']].rename(src_train='src_train1').\
    to_dataframe().groupby('src_train1').size()
x10 = x10.to_xarray()

x9 = 0.5*(x6 + x7 - x10)+x8
x9 = x9.rename('kl').to_dataframe().reset_index()

# %%
(
    p9.ggplot(x9)+
        p9.aes(
            'src_train', 'src_train1', 
            fill='np.log10(kl+1)', label='kl.astype(int)'
        )+
        p9.geom_tile()+
        p9.geom_text()
)


