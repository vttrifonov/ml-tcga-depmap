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
from helpers import config
from svd import SVD
from merge import merge
from common.caching import compose, lazy, XArrayCache, PickleCache

from common.caching import FileCache

class DictCache(FileCache):
    def __init__(self, *default, **elem):
        super().__init__(True, '')
        self.default = default[0] if len(default)>0 else None
        self.elem = elem

    def store(self, data, storage):
        for k, v in data.items():
            s = self.elem.get(k, self.default)
            if s is not None:
                s.store(v, storage/str(k))

    def restore(self, storage):
        data = {}
        for k in storage.glob('*'):
            k = k.name
            s = self.elem.get(k, self.default)
            if s is not None:
                data[k] = s.restore(storage/k)
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
def _():    
    pass    

    # %%
    class _analysis2:
        @compose(property, lazy)
        def storage(self):
            return config.cache / 'playground11' / self.name

        def __init__(self, name, train_split_ratio):
            self._train_split_ratio = train_split_ratio
            self.name = name

        @compose(property, lazy, PickleCache(compressor=None))
        def train_split_ratio(self):
            return self._train_split_ratio        

        @compose(property, lazy, PickleCache(compressor=None))
        def train_split(self):
            rows = merge.crispr.rows
            rows['train'] = ('rows', np.random.random(rows.shape[0])<=self.train_split_ratio)
            return rows
        
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
                    getattr(merge, e)[['data']].\
                        assign(src=lambda x: ('rows', [e]*x.sizes['rows']))
                    for e in ['dm_expr', 'gdc_expr']
                ], dim='rows').rename(data='expr', cols='expr_cols'),
                xa.concat([
                    getattr(merge, e)[['data', 'arm']].\
                        assign(src=lambda x: ('rows', [e]*x.sizes['rows']))
                    for e in ['dm_cnv', 'gdc_cnv']
                ], dim='rows').rename(data='cnv', cols='cnv_cols')
            ], join='inner', compat='override')
            data = data.set_coords('arm')
            for x in ['expr', 'cnv']:
                data[x] = data[x].astype(np.float32)
            data['train'] = self.train_split.train
            data['train'] = data.train.fillna(2).astype(str)
            return data


        @compose(property, lazy, Model1.cache)
        def model1(self):
            train = self.data.sel(rows=self.data.train)
            return Model1().fit(train)
        
        @compose(property, lazy, Model2.cache)
        def model2(self):
            train = self.data.sel(rows=self.data.train)
            return Model2().fit(train)
        
        @compose(property, lazy)
        def model3(self):
            train = self.data.sel(rows=self.data.train)
            return Model3().fit(train)
        
    # %%
    self = _analysis2('20230531/0.8', 0.8)

    # %%
    data = self.data
    train = data.sel(rows=data.train)

    # %%
    model = self.model3

    # %%
    model.cnv_cor_plot(train.expr.rename(expr_cols='cols'))

    # %%
    model.cnv_cor_plot(train.crispr.rename(crispr_cols='cols'))

    # %%    
    model.cnv_cor_plot1(train.expr.rename(expr_cols='cols'))

    # %%
    model.cnv_cor_plot1(train.crispr.rename(crispr_cols='cols'))

    # %%
    model.expr_cor(train.crispr.rename(crispr_cols='cols'))

    # %%
    #model1
    crispr1 = xa.merge([
        model.proj.rename('proj'),
        model.proj1.rename('proj1'),
        model.coef[0].rename('coef1'),
        model.coef[1].rename('coef2'),
        model.coef[2].rename('coef3'),
        model.coef[3].rename('coef4'),
        model.coef[4].rename('coef5')
    ], join='inner')

    # %%
    #model2
    crispr1 = xa.merge([
        model.proj.rename('proj'),
        model.proj1.rename('proj1'),
        model.coef[0].rename('coef1'),
        model.coef[1].rename('coef3'),
        model.coef[2].rename('coef4'),
    ], join='inner')

    # %%
    crispr3 = model.predict(data)
    crispr3['data'] = data.crispr
    crispr3['train'] = data.train
    crispr3 = crispr3.persist()
    
    # %%
    x1 = crispr3.sel(rows=crispr3.train)[['cnv', 'expr']]
    x1 = (x1**2).sum(dim='rows')
    x1 = x1.to_dataframe()
    print(x1.mean())
    print(
        p9.ggplot(x1)+
            p9.aes('cnv', 'expr')+
            p9.geom_point(alpha=0.1, size=2)+
            p9.geom_hline(yintercept=0.23)+
            p9.geom_vline(xintercept=0.40)
    )
    import sklearn.metrics as sklm
    print(
        sklm.confusion_matrix(x1.cnv>0.40, x1.expr>0.23)
    )

    # %%
    x1.query('expr>0.3 & cnv>0.4')

    # %%
    x1 = crispr3[['pred', 'data', 'train']]
    x1 = x1.groupby('train').apply(
        lambda x: x[['pred', 'data']].\
            pipe(lambda x: x-x.mean(dim='rows')).\
            pipe(lambda x: x/np.sqrt((x**2).sum(dim='rows'))).\
            pipe(lambda x: (x.pred * x.data).sum(dim='rows'))
    ).rename('cor')
    x1 = x1.to_dataframe()
    x1 = x1.reset_index().pivot_table(
        index='crispr_cols', columns='train', values='cor'
    )
    x1.columns = x1.columns.astype(str)
    print(
        p9.ggplot(x1)+
            p9.aes('False', 'True')+
            p9.geom_point(alpha=0.1, size=2)
    )

    # %%
    x1.sort_values('False')

    # %%
    data1 = self.data1
    crispr3 = model.predict(data1)
    crispr3['train'] = data1.train
    crispr3 = crispr3.persist()

    # %%
    x1 = crispr3[['score', 'train']]
    x1 = x1.to_dataframe()
    (
        p9.ggplot(x1)+p9.aes('train', 'np.clip(score, -20000, 1000)')+
            p9.geom_violin()
    )

    # %%    
    x2 = crispr1.sel(crispr_cols=['ZFYVE16 (9765)']).persist()
    x2 = xa.merge([x2, crispr3], join='inner')

    # %%
    x3 = x2[['data', 'pred', 'train']].to_dataframe()
    print(
        p9.ggplot(x3)+            
            p9.aes('data', 'pred', color='train')+
            p9.geom_point()+
            p9.geom_smooth(method='lm')
    )
    print(
        x3.groupby('train').aggregate(lambda x: x.loc[:, ['data', 'pred']].corr().iloc[0,1])
    )

    # %%
    from scipy.stats import entropy
    x2['coef1'].rename('r').to_dataframe().assign(
        r2=lambda x: x.r**2/(x.r**2).sum()
    ).r2.sort_values().\
        pipe(lambda x: np.exp(entropy(x)))
        #plot(kind='bar')
        #tail(10)    

    # %%
    x2['coef2'].rename('r').to_dataframe().assign(
        r2=lambda x: x.r**2
    ).sort_values('r2').tail(10)

    # %%
    x2['coef3'].rename('r').to_dataframe().assign(
        r2=lambda x: x.r**2
    ).sort_values('r2').tail(10)

