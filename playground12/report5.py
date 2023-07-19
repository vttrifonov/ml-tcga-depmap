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
from . import _analysis, _scale2 as scale

# %%
def dist(x4, d):
    x5, x6 = [
        x4.sel(src_train_pc=x4.src_train==n)
        for n in ['gdc:True', 'dm:True']
    ]
    def _(x5, x6):
        x5, x6 = [
            x.sel(src_train1=x.src_train.data[0]).\
                sel(rows=x.src_train2==x.src_train.data[0]).\
                drop(['src_train', 'src_train1', 'src_train2']).\
                rename(src_train_pc=f'src_train_pc{s}', rows=f'rows{s}')
            for x, s in [(x5, '1'), (x6, '2')]
        ]
        return d(x5, x6)
    x9 = xa.concat([
        xa.concat([
            _(x5, x6).expand_dims(src_train2=[n6])
            for n6, x6 in x4.groupby('src_train')
        ], dim='src_train2').expand_dims(src_train1=[n5])
        for n5, x5 in x4.groupby('src_train')
    ], dim='src_train1')
    x9 = x9.rename('dist').to_dataframe().reset_index()
    return x9

def kl(x5, x6): 
    x8 = x5.s * (x5.v @ x6.v) / x6.s
    x8 = (x8**2).sum()

    x9 = ((x5.m - x6.m) @ x6.v)/x6.s
    x9 = (x9**2).sum()

    x10 = x5.sizes['src_train_pc1']

    x11 = np.log(x6.s).sum()-np.log(x5.s).sum()

    x12 = np.log(2*np.pi)*(x6.sizes['src_train_pc2']-x5.sizes['src_train_pc1'])

    return 0.5*(x12 + x8 + x9 - x10) + x11

def w2(x5, x6):
    x7 = ((x5.m - x6.m)**2).sum(dim='cols1')

    x8 = (x5.s**2).sum()+(x6.s**2).sum()

    x9 = x5.s * (x5.v @ x6.v) * x6.s   
    x9 = da.array.linalg.svd(x9.data)[1].sum()
    
    return x7+x8-2*x9

def log_prob(x5, x6):
    x7 = ((x5-x6.m) @ x6.v)/x6.s
    x7 = (x7**2).sum(dim='src_train_pc2').mean()

    x8 = np.log(2*np.pi)*x6.sizes['src_train_pc2']

    x9 = np.log(x6.s).sum()

    return -0.5*(x8+x7)+x9

def kl1(x5, x6):
    return (-log_prob(x5.data, x6)+log_prob(x6.data, x6))


# %%
class _analysis5(_analysis):    
    def __init__(self, *args, src, perm):
        super().__init__(*args)
        self.src = src
        self.perm = perm

# %%
@compose(property, lazy)
def x1(self):
    x1 = self.src
    data = self.data2
    x1 = data[['src', x1]].rename({x1: 'data', f'{x1}_cols': 'cols'})
    x1['train'] = self._train_split.train

    x2 = {k: list(x.data) for k, x in x1.rows.groupby(x1.src)}
    x2 = x2['dm']+list(np.random.choice(x2['gdc'], len(x2['dm']), replace=False))
    x1 = x1.sel(rows=x1.rows.isin(x2))

    if self.perm:
        x2, x3 = [x for _, x in x1.groupby('src')]
        x2['cols'] = 'cols', np.random.permutation(x2.cols.data)
        x1 = xa.concat([x2, x3], dim='rows')

    x1 = xa.merge([x1.drop('data'), scale(x1.data).rename('data')], join='inner')
    x1 = x1.transpose('rows', 'cols')
    return x1
_analysis5.x1 = x1

# %%
@compose(property, lazy)
def x2(self):
    x1 = self.x1
    
    x2 = SVD.from_mat(x1.data, n=200, solver='rand').us.persist()
    x2 = xa.merge([x2.rename('data'), x1[['src', 'train']]])
    x2['src_train'] = x2.src.to_series()+':'+x2.train.to_series().astype(str)
    x2 = x2.rename(pc='cols1')
    return x2
_analysis5.x2 = x2

# %%
@compose(property, lazy)
def x3(self):
    x2 = self.x2
    
    x3 = x2.groupby('src_train')
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
    return x3
_analysis5.x3 = x3

# %%
@compose(property, lazy)
def x9(self):
    x3 = self.x3
    
    x4 = (x3.v*x3.s)
    x5 = (x3.v/x3.s).rename(pc='pc1', src_train='src_train1')
    x6 = x4 @ x5
    x6 = (x6**2).sum(dim=['pc', 'pc1'])

    x7 = x3.m - x3.m.rename(src_train='src_train1')
    x7 = xa.dot(x7, x5, dims='cols1')
    x7 = (x7**2).sum(dim='pc1')

    x8 = np.log(x3.s).sum(dim='pc')
    x8 = x8.rename(src_train='src_train1')-x8

    x9 = 0.5*(x6 + x7 - x3.sizes['cols1'])+x8
    x9 = x9.rename('kl').to_dataframe().reset_index()
    return x9
_analysis5.x9 = x9

# %%
@compose(property, lazy)
def x3_1(self):
    x2 = self.x2
    x3 = x2.groupby('src_train')
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
    x3['s'] = np.sqrt(x3.s**2+1e-6)
    x3 = x3.persist()
    return x3
_analysis5.x3_1 = x3_1

# %%
@property
def x4_1(self):
    return xa.merge([
        self.x3_1.drop('pc'), 
        self.x2[['data', 'src_train']].rename(src_train='src_train2')
    ])
_analysis5.x4_1 = x4_1

# %%
@compose(property, lazy)
def x9_1(self):
    x4, d = self.x4_1, kl
    return dist(x4, d)

_analysis5.x9_1 = x9_1

# %%
@compose(property, lazy)
def x9_2(self):
    x4, d = self.x4_1, w2
    return dist(x4, d)

_analysis5.x9_2 = x9_2

# %%
@compose(property, lazy)
def x9_3(self):    
    x4, d = self.x4_1, kl1    
    return dist(x4, d)

_analysis5.x9_3 = x9_3

# %%
analysis5 = _analysis5('20230531/0.5', 0.5, src='expr', perm=False)
self = analysis5

# %%
x9 = analysis5.x9_1
(
    p9.ggplot(x9)+
        p9.aes(
            'src_train1', 'src_train2', 
            fill='np.log10(dist+1)', label='dist.astype(int)'
        )+
        p9.geom_tile()+
        p9.geom_text()
)

# %%
x9 = analysis5.x9_2
(
    p9.ggplot(x9)+
        p9.aes(
            'src_train1', 'src_train2', 
            fill='np.log10(dist+1)', label='dist.astype(int)'
        )+
        p9.geom_tile()+
        p9.geom_text()
)

# %%
x9 = analysis5.x9_3
(
    p9.ggplot(x9)+
        p9.aes(
            'src_train1', 'src_train2', 
            fill='np.log10(dist+1)', label='dist.astype(int)'
        )+
        p9.geom_tile()+
        p9.geom_text()
)


