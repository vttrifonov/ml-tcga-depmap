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
self = _analysis('20230531/0.8', 0.8)
data = self.data
train = data.sel(rows=data.train)

# %%
from . import _scale1
from ..svd import SVD

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

proj1 = cnv_svd.u @ expr.data

expr_svd = expr.data - cnv_svd.u @ proj1
expr_svd = SVD.from_mat(expr_svd).inv()
expr_svd = xa.merge([expr_svd.v.rename('u'), expr_svd.us.rename('vs')])
expr_svd = expr_svd.sel(pc=range(100)).persist()
expr_svd = expr_svd.assign(
    src=lambda x: ('pc', ['expr']*x.sizes['pc']),
    src_pc=lambda x: ('pc', ['expr:'+x for x in x.pc.astype(str).data]),
).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')

expr_svd1 = SVD.from_mat(expr.data).inv()
expr_svd1 = xa.merge([expr_svd1.v.rename('u'), expr_svd1.us.rename('vs')])
expr_svd1 = expr_svd1.sel(pc=range(205)).persist()
expr_svd1 = expr_svd1.assign(
    src=lambda x: ('pc', ['expr']*x.sizes['pc']),
    src_pc=lambda x: ('pc', ['expr:'+x for x in x.pc.astype(str).data]),
).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')

proj2 = expr_svd1.u @ cnv.data

cnv_svd1 = cnv.data - expr_svd1.u @ proj2
cnv_svd1 = SVD.from_mat(cnv_svd1).inv()
cnv_svd1 = xa.merge([cnv_svd1.v.rename('u'), cnv_svd1.us.rename('vs')])
cnv_svd1 = cnv_svd1.sel(pc=range(100)).persist()
cnv_svd1 = cnv_svd1.assign(
    src=lambda x: ('pc', ['cnv']*x.sizes['pc']),
    src_pc=lambda x: ('pc', ['cnv:'+x for x in x.pc.astype(str).data]),
).set_coords(['src', 'src_pc']).swap_dims(pc='src_pc')

# %%
x = xa.concat([cnv_svd.u, expr_svd.u], dim='src_pc').transpose('rows', 'src_pc')
while True:    
    g = GMM(2).fit(x)
    s = g._fit.p.sum(dim='rows')
    print(s.data)
    if np.all(s>100):
        break

# %%
data1 = self.data1

# %%
test = data1

cnv1 = (test.cnv-cnv.center)/cnv.scale
expr1 = (test.expr-expr.center)/expr.scale

cnv2 = cnv1 @ cnv_svd.vs
expr2 = expr1 @ expr_svd1.vs

expr3 = expr1 - proj1 @ cnv2
expr3 @= expr_svd.vs

cnv3 = cnv1 - proj2 @ expr2
cnv3 @= cnv_svd1.vs

# %%
test1 = xa.concat([cnv2, expr3], dim='src_pc').persist().rename('x').to_dataset()
test1['log_score'], test1['log_proba'] = g.log_proba(test1.x)
test1 = test1.persist()

# %%
x1 = xa.merge([test1.log_score, test.train]).to_dataframe().reset_index()
(
    p9.ggplot(x1)+p9.aes('train', 'np.clip(log_score, -20000, 1000)')+
        p9.geom_violin()
)

# %%
x = xa.concat([cnv_svd.u], dim='src_pc').transpose('rows', 'src_pc')
g1 = GMM(1).fit(x)

# %%
test2 = xa.concat([cnv2], dim='src_pc').persist().rename('x').to_dataset()
test2['log_score'], test2['log_proba'] = g1.log_proba(test2.x)
test2 = test2.persist()

# %%
x1 = xa.merge([test2.log_score, test.train]).to_dataframe().reset_index()
(
    p9.ggplot(x1)+p9.aes('train', 'np.clip(log_score, -20000, 1000)')+
        p9.geom_violin()
)

# %%
x = xa.concat([expr_svd1.u], dim='src_pc').transpose('rows', 'src_pc')
g2 = GMM(1).fit(x)

# %%
test3 = xa.concat([expr2], dim='src_pc').persist().rename('x').to_dataset()
test3['log_score'], test3['log_proba'] = g2.log_proba(test3.x)
test3 = test3.persist()

# %%
x1 = xa.merge([test3.log_score, test.train]).to_dataframe().reset_index()
(
    p9.ggplot(x1)+p9.aes('train', 'np.clip(log_score, -20000, 1000)')+
        p9.geom_violin()
)

# %%
x = xa.concat([expr_svd1.u, cnv_svd1.u], dim='src_pc').transpose('rows', 'src_pc')
while True:    
    g3 = GMM(1).fit(x)
    s = g3._fit.p.sum(dim='rows')
    print(s.data)
    if np.all(s>100):
        break

# %%
test4 = xa.concat([expr2, cnv3], dim='src_pc').persist().rename('x').to_dataset()
test4['log_score'], test4['log_proba'] = g1.log_proba(test2.x)
test4 = test4.persist()

# %%
x1 = xa.merge([test4.log_score, test.train]).to_dataframe().reset_index()
(
    p9.ggplot(x1)+p9.aes('train', 'np.clip(log_score, -20000, 1000)')+
        p9.geom_violin()
)


