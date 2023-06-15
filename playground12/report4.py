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
from ..common.caching import compose, lazy

class GMM:
    def __init__(self, k):
        self.k = k

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
    def n_features(self):
        return self._fit.sizes[self._dims[1]]

    @compose(property, lazy)
    def log_det(self):
        return np.log(self._fit.s).sum(dim='pc')
    
    @compose(property, lazy)
    def norm_factor(self):
        return -0.5*np.log(2*np.pi)*self.n_features+self.log_det
    
    def log_proba(self, x):
        x = x - self._fit.m
        x = xa.dot(x, self.vs, dims=self._dims[1])
        log_proba = self.norm_factor - 0.5*(x**2).sum(dim='pc')
        m = log_proba.max(dim='clust')
        log_score = np.log(np.exp(log_proba - m).sum(dim='clust'))
        log_proba -= log_score
        log_score += m
        return log_score, log_proba

# %%
from . import _scale1
from ..svd import SVD
from types import SimpleNamespace as namespace

class _analysis1:
    def __init__(self, prev, name):
        self.prev = prev
        self.name = name

    @compose(property, lazy)
    def storage(self):
        return self.prev.storage/self.name
    
    def fit(self, x1, x2=None):
        x3 = [x1.data]
        if x2:
            x3 = x3 + [x2.data]
        x3 = [
            _scale1(x).rename('data').reset_coords(['center', 'scale'])
            for x in x3
        ]

        svd1 = SVD.from_mat(x3[0].data).inv()
        svd1 = xa.merge([svd1.v.rename('u'), svd1.us.rename('vs')])
        svd1 = svd1.sel(pc=range(x1.pc)).persist()
        svd1 = svd1.assign(
            src=lambda x: ('pc', [x1.src]*x.sizes['pc']),
            src_pc=lambda x: ('pc', [x1.src+':'+x for x in x.pc.astype(str).data]),
        ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
        svd1 = xa.merge([svd1, x3[0].drop('data')])

        self.svd1 = svd1

        if x2:
            proj = svd1.u @ x3[1].data
            svd2 = x3[1].data - svd1.u @ proj
            svd2 = SVD.from_mat(svd2).inv()
            svd2 = xa.merge([svd2.v.rename('u'), svd2.us.rename('vs')])
            svd2 = svd2.sel(pc=range(x2.pc)).persist()
            svd2 = svd2.assign(
                src=lambda x: ('pc', [x2.src]*x.sizes['pc']),
                src_pc=lambda x: ('pc', [x2.src+':'+x for x in x.pc.astype(str).data]),
            ).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')
            svd2['proj'] = proj
            svd2 = xa.merge([svd2, x3[1].drop('data')])
            self.svd2 = svd2

        return self

    def proj(self, x1, x2 = None):
        x1 = (x1 - self.svd1.center)/self.svd1.scale
        x1 @= self.svd1.vs
        x3 = [x1]

        if hasattr(self, 'svd2'):
            x2 = (x2 - self.svd2.center)/self.svd2.scale
            x2 -= self.svd2.proj @ x1
            x2 @= self.svd2.vs
            x3 += [x2]

        x3 = xa.concat(x3, dim='src_pc')
        return x3
            
        
    def gmm(self, k, min_s):
        x = self.svd1.u
        if hasattr(self, 'svd2'):
            x = xa.concat([x, self.svd2.u], dim='src_pc').transpose('rows', 'src_pc')
        while True:    
            g = GMM(k).fit(x)
            s = g._fit.p.sum(dim='rows')
            print(s.data)
            if np.all(s>min_s):
                break
        return g


# %%
self = _analysis('20230531/0.8', 0.8)
data = self.data2.persist()

# %%
train = data.sel(rows=data.train).drop('train')
train = train.sel(rows=train.src=='dm').drop('src')
train = train.sel(crispr_rows=train.rows).drop('crispr_rows')
train = train.persist()

a1, a2, a3, a4 = [
    _analysis1(self, n).fit(*(
        namespace(
            data=train[src],
            src=src,
            pc=pc
        )
        for src, pc in a
    ))
    for n, a in [
        ('a1', [('cnv', 205), ('expr', 100)]),
        ('a2', [('expr', 205), ('cnv', 100)]),
        ('a3', [('cnv', 205)]),
        ('a4', [('expr', 205)])
    ]
]

# %%
g, g1, g2, g3 = [
    a.gmm(*x)
    for a, x in [
        (a1, (2, 100)), 
        (a3, (1, 0)),
        (a4, (1, 0)),
        (a2, (1, 100))
    ]
]

# %%
test = data
test['train1'] = xa.where(
    test.src=='gdc', 
    'gdc', 
    test[['src', 'train']].to_dataframe().pipe(lambda x: x.src+':'+x.train.astype(str)).to_xarray()
)

# %%

test1 = a1.proj(test.cnv, test.expr).persist().rename('x').to_dataset()
test1['log_score'], test1['log_proba'] = g.log_proba(test1.x)
test1 = test1.persist()

# %%
x1 = xa.merge([test1.log_score, test.train1]).to_dataframe().reset_index()
(
    p9.ggplot(x1)+p9.aes('train1', 'np.clip(log_score, -20000, 1000)')+
        p9.geom_violin()
)

# %%
test2 = a3.proj(test.cnv).persist().rename('x').to_dataset()
test2['log_score'], test2['log_proba'] = g1.log_proba(test2.x)
test2 = test2.persist()

# %%
x1 = xa.merge([test2.log_score, test.train1]).to_dataframe().reset_index()
(
    p9.ggplot(x1)+p9.aes('train1', 'np.clip(log_score, -20000, 1000)')+
        p9.geom_violin()
)

# %%
test3 = a4.proj(test.expr).persist().rename('x').to_dataset()
test3['log_score'], test3['log_proba'] = g2.log_proba(test3.x)
test3 = test3.persist()

# %%
x1 = xa.merge([test3.log_score, test.train1]).to_dataframe().reset_index()
(
    p9.ggplot(x1)+p9.aes('train1', 'np.clip(log_score, -20000, 1000)')+
        p9.geom_violin()
)

# %%
test4 = a2.proj(test.expr, test.cnv).persist().rename('x').to_dataset()
test4['log_score'], test4['log_proba'] = g3.log_proba(test2.x)
test4 = test4.persist()

# %%
x1 = xa.merge([test4.log_score, test.train1]).to_dataframe().reset_index()
(
    p9.ggplot(x1)+p9.aes('train1', 'np.clip(log_score, -20000, 1000)')+
        p9.geom_violin()
)


