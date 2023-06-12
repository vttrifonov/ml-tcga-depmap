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
self = _analysis('20230531/0.8', 0.8)

# %%
data = self.data
train = data.sel(rows=data.train)

# %%
model = self.model3

# %%
crispr3 = model.predict(data)
crispr3['data'] = data.crispr
crispr3['train'] = data.train
crispr3 = crispr3.persist()

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
data1 = self.data1
crispr4 = model.predict(data1)
crispr4['train'] = data1.train
crispr4 = crispr4.persist()

# %%
x1 = crispr4[['score', 'train']]
x1 = x1.to_dataframe()
(
    p9.ggplot(x1)+p9.aes('train', 'np.clip(score, -20000, 1000)')+
        p9.geom_violin()
)



