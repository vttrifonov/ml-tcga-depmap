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

from ..common.caching import compose, lazy
from . import _model4, _analysis

# %%
def cor(x, y):
    c = xa.merge(
        [x.rename('x'), y.rename('y')], 
        join='inner'
    )
    c -= c.mean(dim='rows')
    c /= np.sqrt((c**2).sum(dim='rows'))
    c = xa.dot(c.x, c.y, dims='rows')
    return c

class _test:
    def __init__(self, prev, a):
        self.a = prev.model(a)
        self.t = prev.test
        self.p = xa.Dataset(dict(zip(
            ('log_score', 'pred'), 
            self.a.predict(*[self.t[s] for s, _ in prev.a[a][1]])
        )))

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
        ], join='inner')        
        x2 = x2.groupby('train1').apply(lambda x: cor(x.data, x.pred))
        x2 = x2.rename('cor').to_dataframe().reset_index()
        x2 = x2.pivot_table(index='crispr_cols', columns='train1', values='cor')
        return x2

@compose(lazy)
def _model4_test(self, a):
    return _test(self, a)

_model4.test1 = _model4_test

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
analysis = _analysis('20230531/0.8', 0.8)

# %%
a1, a2, a3, a4 = (analysis.model4.test1(a) for a in ['a1', 'a2', 'a3', 'a4'])

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


