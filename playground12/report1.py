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
model = self.model1

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
crispr3 = model.predict(data)
crispr3['data'] = data.crispr
crispr3['train'] = data.train
crispr3 = crispr3.persist()

# %%
x1 = crispr3.sel(rows=crispr3.train)[['cnv', 'expr']]
x1 = (x1**2).sum(dim='rows')/crispr3.train.sum()
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


