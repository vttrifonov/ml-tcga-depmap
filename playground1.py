import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common.defs import pipe, lapply, lfilter
import seaborn as sb
import more_itertools as mit
import functools as ft
from analysis1 import analysis1 as a
from expr import expr
import ae

import gc
gc.collect()

import sklearn.decomposition as skld
import sklearn.preprocessing as sklp
import xarray as xa

x1 = xa.Dataset()

x1['mat'] = expr.full_mat.xarray
x1.mat.values = np.nan_to_num(x1.mat.values).astype(np.float32)

x1 = x1.merge(expr.genes.groupby('Ensembl_Id').agg(sum).rename_axis('col'))
x1['col_mean'] = x1.s/x1.n
x1 = x1.sel({'col': x1['col_mean']>0})

x1['col_var'] = pd.Series(x1.mat.values.var(axis=0), index=x1.col)
x1 = x1.sel({'col': x1['col_var']>1e-10})
x1['row_var'] = x1.mat.var(axis=1)
x1 = x1.sel({'row': x1['row_var']>1e-10})
x1['row_sel'] = pd.Series(np.random.random(x1.mat.shape[0])<0.7, index=x1.row)
x2 = skld.PCA(3000).fit(x1.mat[x1.row_sel,:])

x1['tr_mat'] = xa.DataArray(
    x2.inverse_transform(x2.transform(x1.mat)),
    coords=x1.mat.coords, dims=x1.mat.dims
)

x3_1 = sklp.StandardScaler()
x1['cor'] = pd.Series(
    (x3_1.fit_transform(x1.mat.T)*x3_1.fit_transform(x1.tr_mat.T)).mean(axis=0),
    index=x1.row
).to_xarray()

sb.histplot(x='cor', hue='row_sel', data=x1)

sb.scatterplot(
    x=x1.mat[np.where(~x1.row_sel)[0][1],:],
    y=x1.tr_mat[np.where(~x1.row_sel)[0][1],:]
)

files = expr.files
#files = files[files.project_id == 'TCGA-COAD'].id.reset_index(drop=True)
files = files.id.reset_index(drop=True)

d = a.expr_data(files)
d.m = {}

d.m['pca'] = d.fit(ae.PCA(5000))
d.m['pca'].ae.model.fit(mit.first(d.train.batch(sum(d.select)))[0])

d.m['ae1'] = d.fit(ae.AE(len(d.mat.colnames), 100, 'linear', 'linear', 'adam', 'mse'))
d.m["ae1"].ae.model.fit(
    d.train.batch(sum(d.select)).repeat(),
    validation_data=d.test.batch(sum(~d.select)), validation_steps=1,
    epochs=100, steps_per_epoch=1
)

np.array([((x[0]-x[1])**2).mean() for x in d.m['ae1'].train_decoded]).mean()

x = [np.corrcoef(y[0], y[1])[0,1]**2 for y in  d.m['pca'].test_decoded]