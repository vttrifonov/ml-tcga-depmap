# %%
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9
from types import SimpleNamespace as namespace
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
import dask
from uuid import uuid4 as uuid
from weakref import ref

from ..svd import SVD
from ..merge import merge
from ..helpers import config
from ..common.caching import compose, lazy, XArrayCache, PickleCache
from ..common.caching import FileCache

class DictCache(FileCache):
    def __init__(self, *default, **elem):
        super().__init__(True, '')        
        self.elem = {k: (i, v) for i, (k, v) in enumerate(elem.items())}
        self.default = default[0] if len(default)>0 else None
        self.default = (len(elem), self.default)

    def store(self, data, storage):
        for k, v in data.items():
            i, s = self.elem.get(k, self.default)            
            if s is not None:
                s.store(v, storage/f'{i}.{k}')
        
    def restore(self, storage):
        data = {}
        ks = (k.name.split('.') for k in storage.glob('*'))
        ks = ((int(i), k) for i, k in ks)
        for _, k in sorted(ks):
            i, s = self.elem.get(k, self.default)
            if s is not None:
                data[k] = s.restore(storage/f'{i}.{k}')
        return data

class ArrayCache(FileCache):
    def __init__(self, elem):
        super().__init__(True, ext='')
        self.elem = elem

    def store(self, data, storage):
        for i, v in enumerate(data):
            self.elem.store(v, storage/str(i))

    def restore(self, storage):
        i = [int(x.name) for x in storage.glob('*')]
        return [
            self.elem.restore(storage/str(i))
            for i in sorted(i)
        ]
    
class ClassCache(DictCache):
    def __init__(self, cls, **elem):
        super().__init__(**elem)
        self.cls = cls

    def store(self, data, storage):
        super().store(data.__dict__, storage)

    def restore(self, storage):
        return self.cls(**super().restore(storage))
    
# %%

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

config.exec()

storage = config.cache/'playground12'

# %%
def _scale1(d, rows=['rows']):
    cols = list(set(d.dims)-set(rows))
    d['center'] = (cols, d.mean(dim=rows))
    d = d - d.center
    d['scale'] = (cols, np.sqrt((d**2).mean(dim=rows)))
    d = d/d.scale
    return d

def _scale2(d, rows=['rows']):
    cols = list(set(d.dims)-set(rows))
    d['center'] = (cols, d.mean(dim=rows).data)
    d = d - d.center
    d['scale'] = (cols, np.sqrt((d**2).sum(dim=rows)).data)
    d = d/d.scale
    return d

def gmm(x1, k):
    from sklearn.mixture import GaussianMixture

    d = [x1[d] for d in x1.dims]
    x2 = GaussianMixture(k, verbose=2).fit(x1.data)
    x3 = xa.Dataset()
    x3['m'] = xa.DataArray(x2.means_, [('clust', range(x2.n_components)), d[1]])
    x3['p'] = xa.DataArray(x2.predict_proba(x1.data), [d[0], x3.clust])

    u = x1-x3.m
    w = x3.p/x3.p.sum(dim=d[0].name)
    u = np.sqrt(w) * u
    u = u.transpose('clust', *x1.dims)
    u, s, vt = np.linalg.svd(u.values, full_matrices=False)
    x3['s'] = xa.DataArray(s, [x3.clust, ('pc', range(s.shape[1]))])
    x3['u'] = xa.DataArray(u, [x3.clust, d[0], x3.pc])
    x3['v'] = xa.DataArray(vt, [x3.clust, x3.pc, d[1]])

    return x3

def log_proba(x, vs, cols, pcs):
    p1 = xa.dot(x, vs, dims=cols)
    p2 = np.log(2*np.pi)*vs.sizes[cols]
    p3 = (vs**2).sum(dim=cols)
    p3 = np.log(p3).sum(dim=pcs)
    p1 = (p1**2).sum(dim=pcs)
    p1 = -0.5*(p2-p3+p1)
    return p1

# %%
def _():
    x1 = (2*np.random.rand(9)-1).reshape((3,3))
    x1 = (x1+x1.T)/2
    x1 -= np.diag(np.diag(x1))
    x1 += np.eye(len(x1))
    print(
        x1, 
        np.linalg.eigvalsh(x1), 
        sep='\n'
    )
    x1 = np.linalg.cholesky(x1)
    x2 = np.random.randn(100*len(x1)).reshape((-1,len(x1)))
    x2 -= x2.mean(axis=0)
    x2, _ = np.linalg.qr(x2)
    x2 = x2 @ x1.T
    print(x2.T @ x2)

# %%
class Model1:
    def fit(self, train):
        crispr, expr, cnv = [
            _scale1(x).persist().rename('data').reset_coords(['center', 'scale'])
            for x in [train.crispr, train.expr, train.cnv]
        ]

        cnv_svd = [
            (a, SVD.from_mat(x.data).inv())
            for a, x in cnv.groupby('arm')
        ]
        cnv_svd = {
            a: xa.merge([x.v.rename('u'), x.us.rename('vs')]).\
            pipe(lambda x: x.sel(pc=range(5))).\
            persist().\
            assign(
                arm=lambda x: ('pc', [a]*x.sizes['pc']),
                arm_pc=lambda x: ('pc', a+':'+x.pc.astype(str).to_series())
            ).set_coords('arm').swap_dims(pc='arm_pc') 
            for a, x in cnv_svd
        }
        cnv_svd = namespace(
            u=xa.concat([x.u for x in cnv_svd.values()], dim='arm_pc'),
            vs={a: x.vs for a, x in cnv_svd.items()}
        )
        self.cnv_svd = cnv_svd
    
        cnv_svd1 = SVD.from_mat(cnv_svd.u).inv()
        cnv_svd1 = xa.merge([cnv_svd1.v.rename('u'), cnv_svd1.us.rename('vs')]).persist()
        cnv_svd1 = cnv_svd1.assign(
            src=lambda x: ('pc', ['cnv']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['cnv:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        self.cnv_svd1 = cnv_svd1

        expr1_svd = expr.data - cnv_svd1.u @ (cnv_svd1.u @ expr.data)
        expr1_svd = SVD.from_mat(expr1_svd).inv()
        expr1_svd = xa.merge([expr1_svd.v.rename('u'), expr1_svd.us.rename('vs')]).persist()
        expr1_svd = expr1_svd.sel(pc=range(100))
        expr1_svd = expr1_svd.assign(
            src=lambda x: ('pc', ['expr']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['expr:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        self.expr1_svd = expr1_svd

        u = xa.concat([cnv_svd1.u, expr1_svd.u], dim='src_pc')

        self.proj = (u @ crispr.data).persist()
        self.proj1 = (cnv_svd1.u @ expr.data).persist()

        self.expr = expr.drop('data')
        self.cnv = cnv.drop('data')
        self.crispr = crispr.drop('data')

        return self
    
    @compose(property, lazy)
    def coef(self):                
        coef1 = self.proj @ self.cnv_svd1.vs

        coef2 = xa.concat([
            coef1 @ x
            for x in self.cnv_svd.vs.values()
        ], dim='cnv_cols')

        coef3 = self.proj @ self.expr1_svd.vs

        coef4 = self.proj1 @ self.cnv_svd1.vs

        coef5 = xa.concat([
            coef4 @ x
            for x in self.cnv_svd.vs.values()
        ], dim='cnv_cols')

        return (coef1, coef2, coef3, coef4, coef5)

    def cnv_cor_plot(self, x):
        cnv_cor = self.cnv_svd.u @ _scale1(x)
        cnv_cor['col_arm'] = self.cnv.arm.rename(cnv_cols='cols').drop('arm')
        cnv_cor = (cnv_cor**2).compute().rename('r2')
        cnv_cor = cnv_cor.to_dataframe().sort_values('r2')
        cnv_cor['f'] = cnv_cor.arm==cnv_cor.col_arm
        sns.boxplot(
            x='f',
            y='r2',
            data=cnv_cor[cnv_cor.pc==0]
        )

    def cnv_cor_plot1(self, x):
        cnv_cor = self.cnv_svd1.u @ _scale1(x)
        cnv_cor = (cnv_cor**2).rename('r2').to_dataframe().reset_index()
        cnv_cor = cnv_cor.groupby('cols').r2.sum()
        print(cnv_cor.mean())
        sns.histplot(
            x='r2',
            data=cnv_cor.to_frame()
        )
        plt.show()

    def expr_cor(self, x):
        expr_cor = self.expr1_svd.u @ _scale1(x)
        expr_cor = (expr_cor**2).rename('r2').to_dataframe().reset_index()
        expr_cor = expr_cor.groupby('cols').r2.sum()
        print(expr_cor.mean())
        sns.histplot(
            x='r2',
            data=expr_cor.to_frame()
        )
        plt.show()
    
    def predict(self, test):
        cnv1 = ((test.cnv-self.cnv.center)/self.cnv.scale).persist()
        expr1 = (test.expr-self.expr.center)/self.expr.scale
        expr1 = (expr1 - self.coef[4] @ cnv1).persist()

        crispr3 = xa.Dataset()
        crispr3['cnv'] = self.coef[1] @ cnv1
        crispr3['expr'] = self.coef[2] @ expr1
        crispr3['pred'] = self.crispr.scale * (crispr3.cnv + crispr3.expr) + self.crispr.center
        return crispr3
    
    @classmethod
    def _from_dict(cls, **elems):
        self = cls()
        for k, v in elems.items():
            setattr(self, k, v)
        return self

    def store(self, storage):
        self.cache.store(self, storage)

    @classmethod
    def restore(cls, storage):
        return cls.cache.restore(storage)
    
Model1.cache = ClassCache(
    Model1._from_dict,
    cnv_svd = ClassCache(
        namespace,
        u = XArrayCache(),
        vs = DictCache(XArrayCache())
    ),
    cnv_svd1 = XArrayCache(),
    expr1_svd = XArrayCache(),
    proj = XArrayCache(),
    proj1 = XArrayCache(),
    expr = XArrayCache(),
    cnv = XArrayCache(),
    crispr = XArrayCache()
)

# %%
class Model2:
    def fit(self, train):
        crispr, expr, cnv = [
            _scale1(x).persist().rename('data').reset_coords(['center', 'scale'])
            for x in [train.crispr, train.expr, train.cnv]
        ]
    
        cnv_svd1 = SVD.from_mat(cnv.data).inv()
        cnv_svd1 = xa.merge([cnv_svd1.v.rename('u'), cnv_svd1.us.rename('vs')])
        cnv_svd1 = cnv_svd1.sel(pc=range(205)).persist()
        cnv_svd1 = cnv_svd1.assign(
            src=lambda x: ('pc', ['cnv']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['cnv:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        self.cnv_svd1 = cnv_svd1

        expr1_svd = expr.data - cnv_svd1.u @ (cnv_svd1.u @ expr.data)
        expr1_svd = SVD.from_mat(expr1_svd).inv()
        expr1_svd = xa.merge([expr1_svd.v.rename('u'), expr1_svd.us.rename('vs')]).persist()
        expr1_svd = expr1_svd.sel(pc=range(100))
        expr1_svd = expr1_svd.assign(
            src=lambda x: ('pc', ['expr']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['expr:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        self.expr1_svd = expr1_svd

        u = xa.concat([cnv_svd1.u, expr1_svd.u], dim='src_pc')

        self.proj = (u @ crispr.data).persist()
        self.proj1 = (cnv_svd1.u @ expr.data).persist()

        self.expr = expr.drop('data')
        self.cnv = cnv.drop('data')
        self.crispr = crispr.drop('data')

        return self
    
    @compose(property, lazy)
    def coef(self):                
        coef1 = self.proj @ self.cnv_svd1.vs
        coef3 = self.proj @ self.expr1_svd.vs
        coef4 = self.proj1 @ self.cnv_svd1.vs

        return (coef1, coef3, coef4)

    def cnv_cor_plot1(self, x):
        cnv_cor = self.cnv_svd1.u @ _scale1(x)
        cnv_cor = (cnv_cor**2).rename('r2').to_dataframe().reset_index()
        cnv_cor = cnv_cor.groupby('cols').r2.sum()
        print(cnv_cor.mean())
        sns.histplot(
            x='r2',
            data=cnv_cor.to_frame()
        )
        plt.show()

    def expr_cor(self, x):
        expr_cor = self.expr1_svd.u @ _scale1(x)
        expr_cor = (expr_cor**2).rename('r2').to_dataframe().reset_index()
        expr_cor = expr_cor.groupby('cols').r2.sum()
        print(expr_cor.mean())
        sns.histplot(
            x='r2',
            data=expr_cor.to_frame()
        )
        plt.show()
    
    def predict(self, test):
        cnv1 = ((test.cnv-self.cnv.center)/self.cnv.scale).persist()
        expr1 = (test.expr-self.expr.center)/self.expr.scale
        expr1 = (expr1 - self.coef[2] @ cnv1).persist()

        crispr3 = xa.Dataset()
        crispr3['cnv'] = self.coef[0] @ cnv1
        crispr3['expr'] = self.coef[1] @ expr1
        crispr3['pred'] = self.crispr.scale * (crispr3.cnv + crispr3.expr) + self.crispr.center
        return crispr3
    
    @classmethod
    def _from_dict(cls, **elems):
        self = cls()
        for k, v in elems.items():
            setattr(self, k, v)
        return self

    def store(self, storage):
        self.cache.store(self, storage)

    @classmethod
    def restore(cls, storage):
        return cls.cache.restore(storage)
    
Model2.cache = ClassCache(
    Model2._from_dict,
    cnv_svd1 = XArrayCache(),
    expr1_svd = XArrayCache(),
    proj = XArrayCache(),
    proj1 = XArrayCache(),
    expr = XArrayCache(),
    cnv = XArrayCache(),
    crispr = XArrayCache()
)

# %%
class Model3:
    def fit(self, train):
        crispr, expr, cnv = [
            _scale1(x).rename('data').reset_coords(['center', 'scale'])
            for x in [train.crispr, train.expr, train.cnv]
        ]

        cnv_svd = SVD.from_mat(cnv.data).inv()
        cnv_svd = xa.merge([cnv_svd.v.rename('u'), cnv_svd.us.rename('vs')])
        cnv_svd = cnv_svd.sel(pc=range(205)).persist()
        cnv_svd = cnv_svd.assign(
            src=lambda x: ('pc', ['cnv']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['cnv:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')

        expr_svd = expr.data - cnv_svd.u @ (cnv_svd.u @ expr.data)
        expr_svd = SVD.from_mat(expr_svd).inv()
        expr_svd = xa.merge([expr_svd.v.rename('u'), expr_svd.us.rename('vs')])
        expr_svd = expr_svd.sel(pc=range(100)).persist()
        expr_svd = expr_svd.assign(
            src=lambda x: ('pc', ['expr']*x.sizes['pc']),
            src_pc=lambda x: ('pc', ['expr:'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')

        while True:
            g = xa.concat([cnv_svd.u, expr_svd.u], dim='src_pc')
            g = gmm(g.transpose('rows', 'src_pc'), 2)
            s = g.p.sum(dim='rows')
            print(s.data)
            if np.all(s>100):
                break
        
        g['s'] = 1/(g.s+1e-3)
        g['vs'] = g.v*g.s
        k = g.sizes['src_pc']     
        log_det = np.log(g.s).sum(dim='pc')
        g['nf'] = -0.5*np.log(2*np.pi)*k+log_det
        g['pu'] = np.sqrt(g.p)*g.u
        
        self.proj = g.pu @ crispr.data
        self.proj1 = cnv_svd.u @ expr.data
        self.gmm = g.drop(['v', 's', 'u', 'p', 'pu'])
        self.cnv_svd = cnv_svd.drop('u')        
        self.expr_svd = expr_svd.drop('u')
        self.expr = expr.drop('data')
        self.cnv = cnv.drop('data')
        self.crispr = crispr.drop('data')

        return self
    
    @compose(property, lazy)
    def coef(self):
        coef1 = xa.dot(self.proj, self.gmm.vs, dims='pc') @ self.cnv_svd.vs
        coef3 = xa.dot(self.proj, self.gmm.vs, dims='pc') @ self.expr_svd.vs
        coef4 = self.proj1 @ self.cnv_svd.vs
        self.coef = (coef1, coef3, coef4)

        return (coef1, coef3, coef4)

    def cnv_cor_plot1(self, x):
        cnv_cor = self.cnv_svd1.u @ _scale1(x)
        cnv_cor = (cnv_cor**2).rename('r2').to_dataframe().reset_index()
        cnv_cor = cnv_cor.groupby('cols').r2.sum()
        print(cnv_cor.mean())
        sns.histplot(
            x='r2',
            data=cnv_cor.to_frame()
        )
        plt.show()

    def expr_cor(self, x):
        expr_cor = self.expr1_svd.u @ _scale1(x)
        expr_cor = (expr_cor**2).rename('r2').to_dataframe().reset_index()
        expr_cor = expr_cor.groupby('cols').r2.sum()
        print(expr_cor.mean())
        sns.histplot(
            x='r2',
            data=expr_cor.to_frame()
        )
        plt.show()
    
    def predict(self, test):
        cnv1 = (test.cnv-self.cnv.center)/self.cnv.scale
        cnv1 @= self.cnv_svd.vs

        expr1 = (test.expr-self.expr.center)/self.expr.scale
        expr1 -= self.proj1 @ cnv1
        expr1 @= self.expr_svd.vs

        pred = xa.concat([cnv1, expr1], dim='src_pc')
        pred = pred - self.gmm.m
        pred = xa.dot(pred, self.gmm.vs, dims='src_pc')
        prob = self.gmm.nf - 0.5*(pred**2).sum(dim='pc')
        m = prob.max(dim='clust')
        prob = np.exp(prob - m)
        score = prob.sum(dim='clust')
        prob /= score
        score = m + np.log(score)

        pred = xa.dot(pred, self.proj, dims='pc')
        pred = xa.dot(pred, np.sqrt(prob), dims='clust')
        pred = self.crispr.scale * pred + self.crispr.center

        crispr3 = xa.Dataset()
        crispr3['pred'] = pred
        crispr3['score'] = score

        return crispr3
    
    @classmethod
    def _from_dict(cls, **elems):
        self = cls()
        for k, v in elems.items():
            setattr(self, k, v)
        return self

    def store(self, storage):
        self.cache.store(self, storage)

    @classmethod
    def restore(cls, storage):
        return cls.cache.restore(storage)
    
Model3.cache = ClassCache(
    Model3._from_dict,
    cnv_svd = XArrayCache(),
    expr_svd = XArrayCache(),
    proj = XArrayCache(),
    proj1 = XArrayCache(),
    gmm = XArrayCache(),
    expr = XArrayCache(),
    cnv = XArrayCache(),
    crispr = XArrayCache()
)

# %%
class _analysis:
    @compose(property, lazy)
    def storage(self):
        return config.cache / 'playground12' / self.name

    def __init__(self, name, train_split_ratio):
        self._train_split_ratio = train_split_ratio
        self.name = name

    @compose(property, lazy, PickleCache(compressor=None))
    def train_split_ratio(self):
        return self._train_split_ratio        

    @compose(property, lazy, XArrayCache())
    def train_split(self):
        rows = xa.concat([
            merge.dm_expr.rows, 
            merge.gdc_expr.rows
        ], dim='rows')
        rows['train'] = ('rows', np.random.random(rows.shape[0])<=self.train_split_ratio)
        return rows.rename('train_split')
    
    @compose(property, lazy)
    def data(self):
        data = xa.merge([
            merge.dm_expr.rename(data='expr', cols='expr_cols'),
            merge.dm_cnv.rename(data='cnv', cols='cnv_cols'),
            merge.crispr.rename(data='crispr', cols='crispr_cols')
        ], join='inner', compat='override')
        data = data.set_coords('arm')
        for x in ['expr', 'cnv', 'crispr']:
            data[x] = data[x].astype(np.float32)
        data['train'] = self.train_split.train
        return data
    
    @compose(property, lazy)
    def data1(self):
        data = xa.merge([
            xa.concat([
                getattr(merge, e)[['data']]
                for e in ['dm_expr', 'gdc_expr']
            ], dim='rows').rename(data='expr', cols='expr_cols'),
            xa.concat([
                getattr(merge, e)[['data']]
                for e in ['dm_cnv', 'gdc_cnv']
            ], dim='rows').rename(data='cnv', cols='cnv_cols')
        ], join='inner', compat='override')
        for x in ['expr', 'cnv']:
            data[x] = data[x].astype(np.float32)
        data['train'] = self.train_split.train
        data['train'] = data.train.where(data.rows.isin(merge.crispr.rows), 2).astype(str)
        data = data.unify_chunks()
        data = data.chunk({
            'rows': 'auto',
            'expr_cols': 'auto',
            'cnv_cols': 'auto'
        })
        data['arm'] = merge.dm_cnv.arm.rename(cols='cnv_cols')
        data = data.set_coords('arm')
        return data
    
    @property
    def data2(self):
        data = self.data1.copy()
        data['crispr'] = merge.crispr.data.rename(rows='crispr_rows', cols='crispr_cols')
        data['src'] = xa.where(data.rows.isin(data.crispr_rows), 'dm', 'gdc')
        data['train'] = self.train_split.train    
        data['train1'] = xa.where(
            data.src=='gdc', 
            'gdc', 
            data[['src', 'train']].to_dataframe().\
                pipe(lambda x: x.src+':'+x.train.astype(str)).to_xarray()
        )    
        return data
    
    @compose(property, lazy)
    def train(self):
        data = self.data2.drop('train1')
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            train = data.sel(rows=data.train).drop('train')
        train = train.sel(rows=train.src=='dm').drop('src')
        train = train.sel(crispr_rows=train.rows).drop('crispr_rows')
        train = train.persist()
        return train

    @compose(property, lazy)
    def model1(self):
        train = self.data.sel(rows=self.data.train)
        return Model1().fit(train)
    
    @compose(property, lazy)
    def model2(self):
        train = self.data.sel(rows=self.data.train)
        return Model2().fit(train)
    
    @compose(property, lazy)
    def model3(self):
        train = self.data.sel(rows=self.data.train)
        return Model3().fit(train)
        

# %%
class FromStorage:
    def __init__(self, *args, __restoring__=False, **kwargs):
        if __restoring__:
            return
        super().__init__(*args, **kwargs)

    @classmethod
    def from_storage(cls, elems):
        self = cls(__restoring__=True)
        for k, v in elems.items():
            setattr(self, k, v)
        return self

class ObjectCache(DictCache):    
    cached_data = {}
    cache_id_cache = PickleCache()

    def __init__(self, cls, **elem):
        super().__init__(
            __cache_id__ = self.cache_id_cache, 
            **elem
        )
        self.cls = cls

    def store(self, data, storage):
        if not hasattr(data, '__cache_id__'):
            data.__cache_id__ = uuid()
            self.cached_data[data.__cache_id__] = ref(data)
        super().store(data.__dict__, storage)

    def restore(self, storage):
        data = self.cls(super().restore(storage))
        self.cached_data[data.__cache_id__] = ref(data)
        return data
        
class ObjectRefCache(PickleCache):
    def store(self, data, storage):
        super().store(data.__cache_id__, storage)

    def restore(self, storage):
        id = super().restore(storage)
        return ObjectCache.cached_data[id]()
    
# %%
class GMM:
    def __init__(self, k = None, __restoring__=False):
        if __restoring__:
            return
        self.k = k

    @classmethod
    def from_storage(cls, elems):
        self = cls(__restoring__=True)
        for k, v in elems.items():
            setattr(self, k, v)
        return self

    def fit(self, x1):
        from sklearn.mixture import GaussianMixture

        d = [x1[d] for d in x1.dims]
        x2 = GaussianMixture(self.k, verbose=2).fit(x1.data)
        x3 = xa.Dataset()
        x3['m'] = xa.DataArray(x2.means_, [('clust', range(x2.n_components)), d[1]])
        x3['p'] = xa.DataArray(x2.predict_proba(x1.data), [d[0], x3.clust])
        x3['w'] = x3.p.sum(dim=d[0].name)

        u = x1-x3.m
        w = x3.p/x3.w
        u = np.sqrt(w) * u
        u = u.transpose('clust', *x1.dims)
        u, s, vt = np.linalg.svd(u.values, full_matrices=False)
        x3['s'] = xa.DataArray(s, [x3.clust, ('pc', range(s.shape[1]))])
        x3['u'] = xa.DataArray(u, [x3.clust, d[0], x3.pc])
        x3['v'] = xa.DataArray(vt, [x3.clust, x3.pc, d[1]])
        x3['s'] = 1/(x3.s+1e-3)
        x3['vs'] = x3.v * x3.s
        x3['pu'] = np.sqrt(x3.p) * x3.u
        
        log_det = np.log(x3.s).sum(dim='pc')
        n_features = x1.shape[1]
        x3['norm_factor'] = -0.5*np.log(2*np.pi)*n_features+log_det

        x3 = x3.transpose('clust', d[0].name, 'pc', d[1].name)

        self._fit = x3.drop(['u', 's', 'v', 'p'])

        return self
    
    def proj(self, x):
        f = self._fit
        x = x - f.m
        x = xa.dot(x, f.vs, dims=f.vs.dims[2])
        return x
    
    def _log_proba(self, proj):
        log_proba = self._fit.norm_factor - 0.5*(proj**2).sum(dim='pc')
        m = log_proba.max(dim='clust')
        log_proba -= m
        log_score = np.log(np.exp(log_proba).sum(dim='clust'))
        log_proba -= log_score
        log_score += m
        return log_score, log_proba

    def log_proba(self, x):
        return self._log_proba(self.proj(x))

    class _solver:
        def __init__(self, gmm = None, __restoring__ = False):
            if __restoring__:
                return
            self.gmm = gmm

        @classmethod
        def from_storage(cls, elems):
            self = cls(__restoring__=True)
            for k, v in elems.items():
                setattr(self, k, v)
            return self

        def fit(self, y):
            self.proj = self.gmm._fit.pu @ y
            return self

        def predict(self, x):
            g = self.gmm
            pred = g.proj(x)
            log_score, prob = g._log_proba(pred)
            prob = np.exp(prob)
            pred = xa.dot(pred, self.proj, dims='pc')
            pred = xa.dot(pred, np.sqrt(prob), dims='clust')
            return log_score, pred

    def solver(self):
        return self._solver(self)

GMM.cache = ObjectCache(
    GMM.from_storage,
    _fit = XArrayCache()
)

GMM._solver.cache = ObjectCache(
    GMM._solver.from_storage,
    gmm = ObjectRefCache(),
    proj = XArrayCache()
)

# %%
class Model4:
    GMM = GMM
    SVD = SVD

    def __init__(self, gmm_params = None, __restoring__ = False):
        if __restoring__:
            return
        self.gmm_params = gmm_params

    @classmethod
    def from_storage(cls, elems):
        self = cls(__restoring__=True)
        for k, v in elems.items():
            setattr(self, k, v)
        return self
    
    def fit(self, y, x1, x2=None):
        x3 = [y, x1.data]
        if x2:
            x3 = x3 + [x2.data]
        x3 = [
            _scale1(x).rename('data').reset_coords(['center', 'scale'])
            for x in x3
        ]
        y, x3 = x3[0], x3[1:]

        svd1 = self.SVD.from_mat(x3[0].data).inv()
        svd1 = xa.merge([svd1.v.rename('u'), svd1.us.rename('vs')])
        svd1 = svd1.sel(pc=range(x1.pc)).persist()
        svd1 = svd1.assign(
            src=lambda x: ('pc', [x1.src]*x.sizes['pc']),
            src_pc=lambda x: ('pc', [x1.src+':'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        svd1 = xa.merge([svd1, x3[0].drop('data')])
        u = svd1.u
        self.svd = [svd1]

        if x2:
            svd1['proj'] = svd1.u @ x3[1].data
            svd2 = x3[1].data - svd1.u @ svd1.proj
            svd2 = self.SVD.from_mat(svd2).inv()
            svd2 = xa.merge([svd2.v.rename('u'), svd2.us.rename('vs')])
            svd2 = svd2.sel(pc=range(x2.pc)).persist()
            svd2 = svd2.assign(
                src=lambda x: ('pc', [x2.src]*x.sizes['pc']),
                src_pc=lambda x: ('pc', [x2.src+':'+x for x in x.pc.astype(str).data]),
            ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
            svd2 = xa.merge([svd2, x3[1].drop('data')])
            u = xa.concat([u, svd2.u], dim='src_pc').transpose('rows', 'src_pc')
            self.svd = self.svd + [svd2]

        k, min_s = self.gmm_params
        while True:    
            g = self.GMM(k).fit(u)
            s = g._fit.w
            print(s.data)
            if np.all(s>min_s):
                break
            
        self.g = g
        self.solver = g.solver().fit(y.data)
        self.y = y.drop('data')
        
        return self

    def predict(self, x1, x2=None):
        svd = self.svd        
        x1 = (x1 - svd[0].center)/self.svd[0].scale
        x1 @= self.svd[0].vs
        x3 = [x1]

        if len(svd)>1:
            x2 = (x2 - self.svd[1].center)/self.svd[1].scale
            x2 -= self.svd[0].proj @ x1
            x2 @= self.svd[1].vs
            x3 += [x2]

        x3 = xa.concat(x3, dim='src_pc')

        log_score, pred = self.solver.predict(x3)
        pred = pred*self.y.scale+self.y.center
        return log_score, pred

Model4.cache = ObjectCache(
    Model4.from_storage,
    svd = ArrayCache(XArrayCache()),
    g = GMM.cache,
    solver = GMM._solver.cache,
    y = XArrayCache()
)

# %%
class _model4: 
    Model = Model4

    @compose(property, lazy)
    def train(self):
        return self.prev.train.persist()

    @compose(property, lazy)
    def test(self):
        return self.prev.data2.persist()

    def __init__(self, prev):
        self.prev = prev
        self.a = {
            'a1': ((1, 100), [('cnv', 205), ('expr', 100)]),
            'a2': ((1, 100), [('expr', 205), ('cnv', 100)]),
            'a3': ((1, 0), [('cnv', 205)]),
            'a4': ((1, 0), [('expr', 205)]),
        }

    @compose(property, lazy)
    def storage(self):
        return self.prev.storage/'model4'

    @compose(lazy, Model4.cache)
    def model(self, a):
        g, a = self.a[a]
        train = self.train
        model = self.Model(g).fit(train.crispr, *(
            namespace(
                data=train[src],
                src=src,
                pc=pc
            )
            for src, pc in a
        ))
        return model

@compose(property, lazy)
def _analysis_model4(self):
    return _model4(self)

_analysis.model4 = _analysis_model4

# %%
class Model5:
    GMM = GMM
    SVD = SVD
    scale = staticmethod(_scale2)

    def __init__(self, gmm_params = None, __restoring__ = False):
        if __restoring__:
            return
        self.gmm_params = gmm_params

    @classmethod
    def from_storage(cls, elems):
        self = cls(__restoring__=True)
        for k, v in elems.items():
            setattr(self, k, v)
        return self
    
    def fit(self, y, x1, x2=None):
        x3 = [y, x1.data]
        if x2:
            x3 = x3 + [x2.data]
        x3 = [
            self.scale(x).rename('data').reset_coords(['center', 'scale'])
            for x in x3
        ]
        y, x3 = x3[0], x3[1:]

        svd1 = self.SVD.from_mat(x3[0].data).xarray
        svd1 = svd1.sel(pc=range(x1.pc)).persist()
        svd1 = svd1.assign(
            src=lambda x: ('pc', [x1.src]*x.sizes['pc']),
            src_pc=lambda x: ('pc', [x1.src+':'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        u1, s1 = svd1.u, svd1.s
        u = u1 * s1
        svd1 = xa.merge([svd1.v, x3[0].drop('data')])
        self.svd = [svd1]

        if x2:
            svd1['proj'] = u1 @ x3[1].data
            svd2 = x3[1].data - u1 @ svd1.proj
            svd1['proj'] = (1/(s1+1e-3)) * svd1.proj
            svd2 = self.SVD.from_mat(svd2).xarray
            svd2 = svd2.sel(pc=range(x2.pc)).persist()
            svd2 = svd2.assign(
                src=lambda x: ('pc', [x2.src]*x.sizes['pc']),
                src_pc=lambda x: ('pc', [x2.src+':'+x for x in x.pc.astype(str).data]),
            ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
            u = xa.concat([
                u, svd2.u * svd2.s
            ], dim='src_pc').transpose('rows', 'src_pc')
            svd2 = xa.merge([
                svd2.v, 
                x3[1].drop('data')
            ])
            self.svd = self.svd + [svd2]

        k, min_s = self.gmm_params
        while True:    
            g = self.GMM(k).fit(u)
            s = g._fit.w
            print(s.data)
            if np.all(s>min_s):
                break
            
        self.g = g
        self.solver = g.solver().fit(y.data)
        self.y = y.drop('data')
        
        return self

    def predict(self, x1, x2=None):
        svd = self.svd        
        x1 = (x1 - svd[0].center)/svd[0].scale
        x1 @= svd[0].v
        x3 = [x1]

        if len(svd)>1:
            x2 = (x2 - svd[1].center)/svd[1].scale
            x2 -= svd[0].proj @ x1
            x2 @= svd[1].v
            x3 += [x2]

        x3 = xa.concat(x3, dim='src_pc')

        log_score, pred = self.solver.predict(x3)
        pred = pred*self.y.scale+self.y.center
        return log_score, pred

Model5.cache = ObjectCache(
    Model5.from_storage,
    svd = ArrayCache(XArrayCache()),
    g = GMM.cache,
    solver = GMM._solver.cache,
    y = XArrayCache()
)

# %%
class _model5: 
    Model = Model5

    @compose(property, lazy)
    def train(self):
        return self.prev.train.persist()

    @compose(property, lazy)
    def test(self):
        return self.prev.data2.persist()

    def __init__(self, prev):
        self.prev = prev
        self.a = {
            'a1': ((1, 100), [('cnv', 205), ('expr', 100)]),
            'a2': ((1, 100), [('expr', 205), ('cnv', 100)]),
            'a3': ((1, 0), [('cnv', 205)]),
            'a4': ((1, 0), [('expr', 205)]),
        }

    @compose(property, lazy)
    def storage(self):
        return self.prev.storage/'model5'

    @compose(lazy, Model5.cache)
    def model(self, a):
        g, a = self.a[a]
        train = self.train
        model = self.Model(g).fit(train.crispr, *(
            namespace(
                data=train[src],
                src=src,
                pc=pc
            )
            for src, pc in a
        ))
        return model

@compose(property, lazy)
def _analysis_model5(self):
    return _model5(self)

_analysis.model5 = _analysis_model5