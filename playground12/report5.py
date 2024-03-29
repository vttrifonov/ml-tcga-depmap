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
def cov1(x):
    x = x.copy()
    x['m'] = x.data.mean(dim='rows')
    x['data'] = x.data-x.m
    x['data'] = x.data/np.sqrt(x.sizes['rows'])
    cov = x.data.rename(cols1='cols1_1') @ x.data
    cov = cov + np.diag([1e-6]*cov.shape[0])
    cov = SVD.from_mat(cov).xarray[['s', 'v']]
    cov['s'] = np.sqrt(cov.s)
    x = xa.merge([x.drop_dims('rows'), cov])
    return x

def cov2(x):
    x1 = x[['data']].copy()
    x1['m'] = x1.data.mean(dim='rows')
    x1['data'] = x1.data-x1.m
    x1['data'] = x1.data/np.sqrt(x1.sizes['rows'])
    cov = x1.data.transpose('rows', 'cols1')
    cov = SVD.from_mat(cov).xarray[['s', 'v']]
    cov = cov.sel(pc=(np.cumsum(cov.s**2)/np.sum(cov.s**2)).compute()<0.95)
    #cov['s'] = np.sqrt(cov.s**2+1e-6)
    return xa.merge([x.data, x1.m, cov])

def cov3(x):
    x1 = x.data
    x2 = x1.mean(dim='rows')
    x1 = x1-x2
    x1 = x1/np.sqrt(x1.sizes['rows'])
    cov = x1.transpose('rows', 'cols1')
    cov = SVD.from_mat(cov).xarray
    cov['u'] = np.sqrt(x1.sizes['rows'])*cov.u
    cov = cov.sel(pc=(np.cumsum(cov.s**2)/np.sum(cov.s**2)).compute()<0.95)
    cov['m'] = x2
    return xa.merge([x.data, cov])

def dist(x4, d):
    x9 = xa.concat([
        xa.concat([
            d(x5, x6).expand_dims(src_train2=[n6])
            for n6, x6 in x4.items()
        ], dim='src_train2').expand_dims(src_train1=[n5])
        for n5, x5 in x4.items()
    ], dim='src_train1')
    x9 = x9.rename('dist').to_dataframe().reset_index()
    return x9

def kl(x5, x6): 
    x5 = x5.rename(pc='pc1')
    x6 = x6.rename(pc='pc2')

    x8 = x5.s * (x5.v @ x6.v) / x6.s
    x8 = (x8**2).sum()

    x9 = ((x5.m - x6.m) @ x6.v)/x6.s
    x9 = (x9**2).sum()

    x10 = x5.sizes['pc1']

    x11 = np.log(x6.s).sum()-np.log(x5.s).sum()

    x12 = np.log(2*np.pi)*(x6.sizes['pc2']-x5.sizes['pc1'])

    return 0.5*(x8 + x9 - x10 + x12) + x11

def w2(x5, x6):
    x5 = x5.rename(pc='pc1')
    x6 = x6.rename(pc='pc2')

    x7 = ((x5.m - x6.m)**2).sum(dim='cols1')

    x8 = (x5.s**2).sum()+(x6.s**2).sum()

    x9 = x5.s * (x5.v @ x6.v) * x6.s   
    x9 = da.array.linalg.svd(x9.data)[1].sum()
    
    return x7+x8-2*x9

def ent(x6):
    return 0.5*x6.sizes['pc']*(np.log(2*np.pi)+1)+np.log(x6.s).sum()

def ce(x5, x6): 
    x5 = x5.rename(pc='pc1')
    x6 = x6.rename(pc='pc2')

    x8 = x5.s * (x5.v @ x6.v) / x6.s
    x8 = (x8**2).sum()

    x9 = ((x5.m - x6.m) @ x6.v)/x6.s
    x9 = (x9**2).sum()

    x11 = np.log(x6.s).sum()

    x12 = np.log(2*np.pi)*x6.sizes['pc2']

    return 0.5*(x8 + x9 + x12) + x11

def ent1(x6):
    return ce(x6, x6)

def log_prob(x5, x6):
    x7 = ((x5-x6.m) @ x6.v)/x6.s
    x7 = (x7**2).sum(dim='pc')

    x8 = np.log(2*np.pi)*x6.sizes['pc']

    x9 = np.log(x6.s).sum()

    return -0.5*(x8+x7)-x9

def ce1(x5, x6):
    return -log_prob(x5.data, x6).mean()

def ent2(x6):
    return ce1(x6, x6)

def kl1(x5, x6):
    return ce1(x5, x6)-ent2(x5)

def kl2(x5, x6):
    return ce(x5, x6)-ent(x5)

def proj1(x5, x6):
    x5 = x5.rename(pc='pc1')
    x6 = x6.rename(pc='pc2')
    y1 = (x5.v @ x6.v) * x5.s 
    y1 = y1.transpose('pc1', 'pc2')
    y1 = SVD.from_mat(y1).xarray
    y1['v'] = y1.v @ x6.v
    y1['m'] = x6.m + x6.v @ (x6.v @ (x5.m-x6.m))
    if 'pc2' in y1.dims:
        y1 = y1.drop('pc2')
    return y1

def proj2(x5, x6):    
    y2 = x6.m + ((x5.data-x6.m) @ x6.v) @ x6.v
    y2 = y2.rename('data').to_dataset()
    y2['m'] = y2.data.mean(dim='rows')
    y3 = y2.data-y2.m
    y3 = y3/np.sqrt(y3.sizes['rows'])
    y3 = y3.transpose('rows', 'cols1')
    y3 = SVD.from_mat(y3).xarray[['s', 'v']]
    y3 = y3.sel(pc=range(min(x5.sizes['pc'], x6.sizes['pc'])))
    y2 = xa.merge([y2, y3])
    return y2

def proj3(x6, x7):
    x9 = np.eye(x6.sizes['pc'])
    x9 = xa.DataArray(x9, [('pc2', range(len(x9))), ('pc1', range(len(x9)))])
    def _(x5):
        nonlocal x6, x9
        x5 = proj1(x5, x6).drop_dims('pc1')
        x6 = proj1(x6, x5)
        x9 = (x9 @ x6.u).rename(pc='pc1')
        x6 = x6.drop_dims('pc1')
        return x5    
    x7 = [_(x5) for x5 in x7]
    x9 = x9.rename(pc2='pc')
    return x6, x9, x7

def dist1(x4):
    x8 = []
    for k2, x6 in x4.items():    
        x6, _, x7 = proj3(x6, x4.values())
        x7 = [
            kl(proj1(x5, x6).drop_dims('pc1'), x6).expand_dims(src_train1=[k1]) 
            for k1, x5 in zip(x4.keys(), x7)
        ]
        x7 = xa.concat(x7, dim='src_train1')
        x8.append(x7.expand_dims(src_train2=[k2]))
    x8 = xa.concat(x8, dim='src_train2')
    x8 = x8.rename('dist').rename('dist').to_dataframe().reset_index()
    return x8

def proj4(x6, x7):
    def _(x5):
        nonlocal x6
        x5 = proj2(x5, x6)
        x6 = proj2(x6, x5)
        return x5    
    x7 = [_(x5) for x5 in x7]
    return x6, x7

def dist2(x4):
    x4 = {k: proj2(x, x) for k, x in x4.items()}
    x8 = []
    for k2, x6 in x4.items():    
        x6, x7 = proj4(x6, x4.values())
        x7 = [
            kl1(proj2(x5, x6), x6).expand_dims(src_train1=[k1]) 
            for k1, x5 in zip(x4.keys(), x7)
        ]
        x7 = xa.concat(x7, dim='src_train1')
        x8.append(x7.expand_dims(src_train2=[k2]))
    x8 = xa.concat(x8, dim='src_train2')
    x8 = x8.rename('dist').rename('dist').to_dataframe().reset_index()
    return x8

def proj5(x5, x6):
    x5 = x5.rename(pc='pc1')
    x6 = x6.rename(pc='pc2')
    y1 = (x5.v @ x6.v) * x5.s 
    y1 = y1.transpose('pc1', 'pc2')
    y1 = SVD.from_mat(y1).xarray
    y1['u'] = y1.u @ x5.u
    y1['v'] = y1.v @ x6.v
    y1['m'] = x6.m + x6.v @ (x6.v @ (x5.m-x6.m))
    y1 = y1.drop(['pc1', 'pc2'])
    return y1

def log_prob1(x5, x6):
    x5 = x5.rename(pc='pc1')

    x7 = x5.u @ (x5.s*((x5.v @ x6.v)/x6.s))
    x7 = x7 + ((x5.m-x6.m) @ x6.v)/x6.s
    x7 = (x7**2).sum(dim='pc')

    x8 = np.log(2*np.pi)*x6.sizes['pc']

    x9 = np.log(x6.s).sum()

    return -0.5*(x8+x7)-x9


def ce2(x5, x6):
    return -log_prob1(x5, x6).mean()

def ent3(x6):
    return ce2(x6, x6)

def kl3(x5, x6):
    return ce2(x5, x6)-ent3(x5)


def proj6(x6, x7):
    def _(x5):
        nonlocal x6
        x5 = proj5(x5, x6)
        x6 = proj5(x6, x5)
        return x5    
    x7 = [_(x5) for x5 in x7]
    return x6, x7

def dist3(x4, kl):
    x8 = []
    for k2, x6 in x4.items():    
        x6, x7 = proj6(x6, x4.values())
        x7 = [
            kl(proj5(x5, x6), x6).expand_dims(src_train1=[k1]) 
            for k1, x5 in zip(x4.keys(), x7)
        ]
        x7 = xa.concat(x7, dim='src_train1')
        x8.append(x7.expand_dims(src_train2=[k2]))
    x8 = xa.concat(x8, dim='src_train2')
    x8 = x8.rename('dist').rename('dist').to_dataframe().reset_index()
    return x8


# %%
class _analysis5(_analysis):    
    def __init__(self, *args, src, perm, equalize):
        super().__init__(*args)
        self.src = src
        self.perm = perm
        self.equalize = equalize

# %%
@compose(property, lazy)
def x1(self):
    x1 = self.src
    data = self.data2
    x1 = data[['src', x1]].rename({x1: 'data', f'{x1}_cols': 'cols'})
    x1['train'] = self._train_split.train
    x1['src_train'] = x1.src.to_series()+':'+x1.train.to_series().astype(str)

    if self.equalize:    
        x2 = {k: list(x.data) for k, x in x1.rows.groupby(x1.src)}
        x2 = x2['dm']+list(np.random.choice(x2['gdc'], len(x2['dm']), replace=False))
        x1 = x1.sel(rows=x1.rows.isin(x2))

    if self.perm is not None:    
        x2 = []
        for _, x in x1.groupby(self.perm):
            x['cols'] = 'cols', np.random.permutation(x.cols.data)
            x2.append(x)            
        x1 = xa.concat(x2, dim='rows')

    return x1
_analysis5.x1 = x1

# %%
@compose(property, lazy)
def x2(self):
    x2 = self.x1.copy()
    x3 = x2.data
    #x3 = x2.data.sel(rows=x2.train)
    x4 = x3.mean(dim='rows')
    x3 = x3-x4
    x5 = np.sqrt((x3**2).sum(dim='rows'))
    x3 = x3/x5
    x3 = x3.transpose('rows', 'cols').persist()
    x3 = SVD.from_mat(x3, n=200, solver='rand').v.persist()
    x2 = xa.merge([
        (((x2.data-x4)/x5)@x3).rename('data', pc='cols1'),
        x2.drop_dims('cols')
    ]).persist()
    x2 = {k: v for k, v in x2.groupby('src_train')}
    return x2

_analysis5.x2 = x2

# %%
@compose(property, lazy)
def x3(self):
    x3 = {
        k: cov3(x).persist()
        for k, x in self.x2.items()
    }
    return x3
_analysis5.x3 = x3

# %%
analysis5 = _analysis5('20230531/0.5', 0.5, src='expr', perm=None, equalize=False)
self = analysis5

# %%
#self.perm = None
#self.src = 'expr'
#self.equalize = True
#del self.__lazy__x1_
#del self.__lazy__x2_
#del self.__lazy__x3_

# %%
def test2(self, k1, k2):    
    x4 = self.x3
    x5, x6 = [x4[k] for k in [k1, k2]]

    y1 = proj5(x5, x6)
    y2 = proj5(x6, y1)
    print([
        f(y1, y2).values  for f in [kl, kl3, kl2]
    ])

    print([
        [
            ((x-y1.v.rename(pc='pc1') @ (y1.v.rename(pc='pc1') @ x))**2).sum().values
            for x in [y2.v, y1.m-y2.m]
        ]
        for y1, y2 in [(y1, y2), (y2, y1)]
    ])


def test1(self, k1, k2):    
    x4 = self.x3
    x5, x6 = [x4[k] for k in [k1, k2]]

    y1 = proj2(x5, x6)
    y2 = proj2(x6, y1)
    print([
        f(y1, y2).values  for f in [kl, kl1, kl2]
    ])

    y1 = proj1(x5, x6).drop_dims('pc1')
    y2 = proj1(x6, y1).drop_dims('pc1')
    print([
        f(y1, y2).values for f in [kl, kl2]
    ])

    print([
        [
            ((x-y1.v.rename(pc='pc1') @ (y1.v.rename(pc='pc1') @ x))**2).sum().values
            for x in [y2.v, y1.m-y2.m]
        ]
        for y1, y2 in [(y1, y2), (y2, y1)]
    ])

#test1(self.x3_2, 'gdc:False', 'dm:True')

# %%
x9 = dist(self.x3, lambda x5, x6: kl2(proj5(x5, x6), x6))
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
x9 = dist1(self.x3)
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
x9 = dist2(self.x3)
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
x9 = dist3(self.x3, kl2)
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
x9 = dist3(self.x3, kl3)
(
    p9.ggplot(x9)+
        p9.aes(
            'src_train1', 'src_train2', 
            fill='np.log10(dist+1)', label='dist.astype(int)'
        )+
        p9.geom_tile()+
        p9.geom_text()
)


