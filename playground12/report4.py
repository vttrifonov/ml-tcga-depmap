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
from . import _analysis

# %%
from ..common.caching import compose, lazy, PickleCache
from . import storage, DictCache
from uuid import uuid4 as uuid
from weakref import ref

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
class A:
    def __init__(self, b):
        self.b = b

    def fit(self, a):
        self.a = a
        return self

class B:
    A = A

    def __init__(self, b):
        self.b = b

    @compose(property, lazy)
    def a(self):
        return self.A(self)

class C:
    B = B
    
    def __init__(self, a, b):
        self.b = self.B(b)
        self.a = self.b.a.fit(a)
        
! rm -rf ../.cache/playground12/c

class C1(FromStorage, C):
    class B(FromStorage, B):
        class A(FromStorage, A):
            pass

C1.B.cache = ObjectCache(
    C1.B.from_storage,
    b = PickleCache()
)

C1.B.A.cache = ObjectCache(
    C1.B.A.from_storage,
    b = ObjectRefCache(),
    a = PickleCache()
)

C1.cache = ObjectCache(
    C1.from_storage,
    b = C1.B.cache,
    a = C1.B.A.cache
)

class D:
    storage = storage/'d'
    @compose(property, lazy, C1.cache)
    def c(self):
        return C1(2, 3)


# %% [markdown]
# 
# c = C1(2, 3)
# print(c.a.a, c.b.b)
# 
# c.cache.store(c, storage/'c')
# 
# ObjectCache.cached = {}
# 
# d = c.cache.restore(storage/'c')
# print(d.a.a, d.b.b)

# %%
from ..common.caching import compose, lazy, XArrayCache, PickleCache

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

        u = x1-x3.m
        w = x3.p/x3.p.sum(dim=d[0].name)
        u = np.sqrt(w) * u
        u = u.transpose('clust', *x1.dims)
        u, s, vt = np.linalg.svd(u.values, full_matrices=False)
        x3['s'] = xa.DataArray(s, [x3.clust, ('pc', range(s.shape[1]))])
        x3['u'] = xa.DataArray(u, [x3.clust, d[0], x3.pc])
        x3['v'] = xa.DataArray(vt, [x3.clust, x3.pc, d[1]])
        x3['s'] = 1/(x3.s+1e-3)
        
        self._fit = x3
        self._dims = tuple(x.name for x in d)

        return self
    
    @compose(property, lazy)
    def vs(self):
        return self._fit.v * self._fit.s

    @compose(property, lazy)
    def pu(self):
        return np.sqrt(self._fit.p) * self._fit.u
    
    @compose(property, lazy)
    def n_features(self):
        return self._fit.sizes[self._dims[1]]

    @compose(property, lazy)
    def log_det(self):
        return np.log(self._fit.s).sum(dim='pc')
    
    @compose(property, lazy)
    def norm_factor(self):
        return -0.5*np.log(2*np.pi)*self.n_features+self.log_det

    def proj(self, x):
        x = x - self._fit.m
        x = xa.dot(x, self.vs, dims=self._dims[1])
        return x
    
    def _log_proba(self, proj):
        log_proba = self.norm_factor - 0.5*(proj**2).sum(dim='pc')
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
            self.proj = self.gmm.pu @ y
            return self

        def predict(self, x):
            gmm = self.gmm
            x = gmm.proj(x)
            log_score, prob = gmm._log_proba(x)
            prob = np.exp(prob)
            x = xa.dot(x, self.proj, dims='pc')
            x = xa.dot(x, np.sqrt(prob), dims='clust')
            return log_score, x

    def solver(self):
        return self._solver(self)

GMM.cache = ObjectCache(
    GMM.from_storage,
    _fit = XArrayCache(),
    _dims = PickleCache()
)

GMM._solver.cache = ObjectCache(
    GMM._solver.from_storage,
    gmm = ObjectRefCache(),
    proj = XArrayCache()
)


# %%
from . import _scale1
from . import ArrayCache
from ..svd import SVD
from types import SimpleNamespace as namespace

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
            proj = svd1.u @ x3[1].data
            svd2 = x3[1].data - svd1.u @ proj
            svd2 = self.SVD.from_mat(svd2).inv()
            svd2 = xa.merge([svd2.v.rename('u'), svd2.us.rename('vs')])
            svd2 = svd2.sel(pc=range(x2.pc)).persist()
            svd2 = svd2.assign(
                src=lambda x: ('pc', [x2.src]*x.sizes['pc']),
                src_pc=lambda x: ('pc', [x2.src+':'+x for x in x.pc.astype(str).data]),
            ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
            svd2['proj'] = proj
            svd2 = xa.merge([svd2, x3[1].drop('data')])
            u = xa.concat([u, svd2.u], dim='src_pc').transpose('rows', 'src_pc')
            self.svd = self.svd + [svd2]

        k, min_s = self.gmm_params
        while True:    
            g = self.GMM(k).fit(u)
            s = g._fit.p.sum(dim='rows')
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
            x2 = (x2 - self.svd[1].center)/self.self.svd[1].scale
            x2 -= self.svd[1].proj @ x1
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
    def data(self):
        x = self.prev.data2.persist().copy()
        x['train1'] = xa.where(
            x.src=='gdc', 
            'gdc', 
            x[['src', 'train']].to_dataframe().pipe(lambda x: x.src+':'+x.train.astype(str)).to_xarray()
        )
        return x

    @compose(property, lazy)
    def train(self):
        data = self.data.drop('train1')
        train = data.sel(rows=data.train).drop('train')
        train = train.sel(rows=train.src=='dm').drop('src')
        train = train.sel(crispr_rows=train.rows).drop('crispr_rows')
        train = train.persist()
        return train

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

    @compose(property, lazy, Model4.cache)
    def a1(self):
        return self.model('a1')

@compose(property, lazy)
def _analysis_model4(self):
    return _model4(self)

_analysis.model4 = _analysis_model4

# %%
class _analysis1(Model4):
    def __init__(self, prev, name, gmm_params):
        self.prev = prev
        self.name = name
        super().__init__(gmm_params)

    @compose(property, lazy)
    def storage(self):
        return self.prev.storage/self.name

# %%
class _test:
    def __init__(self, a, t, s):
        self.a = a
        self.t = t
        self.p = xa.Dataset(dict(zip(
            ('log_score', 'pred'), 
            a.predict(*[t[s] for s in s])
        ))).persist()

    @compose(property, lazy)
    def data1(self):
        x1 = xa.merge([
            self.p.log_score, self.t.train1
        ]).to_dataframe().reset_index()
        return x1

    @compose(property, lazy)
    def data2(self):
        x2 = xa.merge([
            self.p.pred.rename('pred'), 
            self.t.crispr.rename('data').rename(crispr_rows='rows'),
            self.t.train1
        ], join='inner').to_dataframe().reset_index()
        x2 = x2.groupby(['crispr_cols', 'train1']).apply(lambda x: x[['pred', 'data']].corr().iloc[0,1]).rename('cor').reset_index()
        x2 = x2.pivot_table(index='crispr_cols', columns='train1', values='cor')
        return x2

def _analysis1_test(self, test, src):
    return _test(self, test, src)

_analysis1.test = _analysis1_test

# %%
def _test_plot1(self):
    return (
        p9.ggplot(self.data1)+
            p9.aes('train1', 'np.clip(log_score, -20000, 1000)')+
            p9.geom_violin()
    )
_test.plot1 = _test_plot1

def _test_plot2(self):
    return (
        p9.ggplot(self.data2)+
            p9.aes('dm:False', 'dm:True')+
            p9.geom_point(alpha=0.1)
    )
_test.plot2 = _test_plot2

# %%
self = _analysis('20230531/0.8', 0.8)

# %%
self = _analysis('20230531/0.8', 0.8)
data = self.data2.persist()

# %%
train = data.sel(rows=data.train).drop('train')
train = train.sel(rows=train.src=='dm').drop('src')
train = train.sel(crispr_rows=train.rows).drop('crispr_rows')
train = train.persist()

# %%
test = data.copy()
test['train1'] = xa.where(
    test.src=='gdc', 
    'gdc', 
    test[['src', 'train']].to_dataframe().pipe(lambda x: x.src+':'+x.train.astype(str)).to_xarray()
)

# %%
a1, a2, a3, a4 = [
    _analysis1(self, n, g).fit(train.crispr, *(
        namespace(
            data=train[src],
            src=src,
            pc=pc
        )
        for src, pc in a
    )) #.test(test, [src for src, _ in a])
    for n, g, a in [
        ('a1', (1, 100), [('cnv', 205), ('expr', 100)]),
        ('a2', (1, 100), [('expr', 205), ('cnv', 100)]),
        ('a3', (1, 0), [('cnv', 205)]),
        ('a4', (1, 0), [('expr', 205)])
    ]
]

# %%
a1, a2, a3, a4 = [
    _analysis1(self, n, g).fit(train.crispr, *(
        namespace(
            data=train[src],
            src=src,
            pc=pc
        )
        for src, pc in a
    )) #.test(test, [src for src, _ in a])
    for n, g, a in [
        ('a1', (1, 100), [('cnv', 205), ('expr', 100)]),
        ('a2', (1, 100), [('expr', 205), ('cnv', 100)]),
        ('a3', (1, 0), [('cnv', 205)]),
        ('a4', (1, 0), [('expr', 205)])
    ]
]

# %%
test1 = a1.test(test, ['cnv', 'expr'])

# %%
a1.plot1()

# %%
a1.plot2()

# %%
a3.plot1()

# %%
a3.plot2()

# %%
a4.plot1()

# %%
a4.plot2()

# %%
a2.plot1()

# %%
a2.plot2()


