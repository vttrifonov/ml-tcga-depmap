import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import more_itertools as mit
import itertools as it
import functools as ft
from pathlib import Path
import zarr
from joblib import Parallel, delayed
import dask.array as daa
import dask_ml.preprocessing as dmlp
import dask_ml.decomposition as dmld
import xarray as xa
from expr import expr
from types import SimpleNamespace
from common.defs import lazy_property
import types
import tensorflow as tf
import tensorflow.keras as tfk

import expr
importlib.reload(expr)
from expr import expr

self = expr.mat2

x1 = self.xarray
x1['var'] = x1.data.var(axis=0).compute()
x1['mean'] = x1.data.mean(axis=0).compute()

plt.scatter(
    x1['mean'],
    np.log10(x1['var']+1e-10)
)
plt.gca().axvline(x=-7)

class Mat:
    def __init__(self, data):
        self.data = data

    @lazy_property
    def ms(self):
        return (self.data**2).mean().compute()

class Fit:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    @lazy_property
    def fit(self):
        return self.model.fit(self.data.train.data)

    def transform1(self, data):
        return self.fit.transform(data)

    def transform2(self, data):
        return self.fit.inverse_transform(self.fit.transform(data))

    @lazy_property
    def transformed(self):
        return SimpleNamespace(
            train=Mat(self.transform(self.data.train.data)),
            test=Mat(self.transform(self.data.test.data))
        )

def chunk_perm(chunks):
    perm = np.cumsum(chunks)
    perm = zip(np.hstack([0, perm[:-1]]), perm)
    perm = [np.random.permutation(np.arange(x[0], x[1])) for x in perm]
    perm = np.hstack(perm)
    return perm

#

data = x1.sel(cols=x1['mean']>(-7))
data['train'] = ('rows', np.random.random(data.rows.shape)<0.7)
data = SimpleNamespace(
    train=Mat(data.sel(rows=data.train).data.data),
    test=Mat(data.sel(rows=~data.train).data.data)
)
normalize = Fit(
    dmlp.StandardScaler(),
    data
)
normalize.transform = normalize.transform1
pca = Fit(
    dmld.PCA(n_components=3000),
    normalize.transformed
)
pca.transform = pca.transform2
pca.transformed.train.ms/pca.data.train.ms
pca.transformed.test.ms/pca.data.test.ms

#

x3_1 = self.col_go[['col', 'display_label']].\
    drop_duplicates().\
    set_index('display_label').col
x3_2 = x3_1.reset_index().display_label.value_counts()
x3_2 = x3_2[(x3_2>=10) & (x3_2<=700)].sort_values()
x3_2

x4_1 = x1.sel(cols=x1['mean']>(-7))
x4_1['data'].data = dmlp.StandardScaler().fit_transform(x4_1.data.data)
x4_2 = x4_1.cols.values
x4_2 = x3_1[x3_1.isin(x4_2)]

x5_5 = set(x4_2['GO:0032886'])
x5_3 = [x for x in x4_1.cols.values if x not in x5_5]
x5_3 = x4_1.sel(cols=x5_3).data.data
x5_1 = x4_1.sel(cols=list(x5_5)).data.data
x5_1 = x5_1.rechunk((x5_1.chunks[0], x5_1.shape[1]))
#x5_1 = x5_1[chunk_perm(x5_1.chunks[0]),:]
x5_2 = daa.linalg.qr(x5_1)
x5_3 = x5_1 @ daa.linalg.solve(x5_2[1], x5_2[0].T @ x5_3)
(x5_3**2).mean().compute()

#

x3_1 = self.col_go[['col', 'display_label']].\
    drop_duplicates().\
    set_index('display_label').col
x3_2 = x3_1.reset_index().display_label.value_counts()
x3_2 = x3_2[(x3_2>=10) & (x3_2<=700)].sort_values()
x3_2 = x3_2.reset_index().reset_index().set_index('index').rename(columns={'level_0': 'i'})

x4_1 = x1.sel(cols=x1['mean']>(-7))
x4_2 = x4_1.cols.values
x4_2 = x3_1[x3_1.isin(x4_2)].reset_index().set_index('display_label')
x4_2 = x4_2.join(x3_2, how='inner')
x4_2 = x4_2.reset_index().set_index('col')
x4_2 = x4_2.join(pd.Series(range(x4_1.cols.shape[0]), index=x4_1.cols.values, name='j'))
x4_1 = x4_1.data.data
x4_1 = dmlp.StandardScaler().fit_transform(x4_1)

x5_1 = tfk.layers.Input((x4_1.shape[1],))
x5_2 = tf.SparseTensor(
    indices=np.array(x4_2[['i', 'j']]),
    values = tf.Variable(
        initial_value = [0]*x4_2.shape[0],
        dtype='float32'
    ),
    dense_shape=(x3_2.shape[0], x4_1.shape[1])
)
x5_2 = tf.sparse.sparse_dense_matmul(x5_2, tf.transpose(x5_1))
x5_2 = tf.transpose(x5_2)
x5_2 = tfk.layers.Dense(x4_1.shape[1])(x5_2)
x5_3 = tfk.Model(inputs=x5_1, outputs=x5_2)
x5_3.compile(optimizer='adam', loss='mean_squared_error')

x5_4 = x4_1[:100,:].compute()
x5_4 = tf.data.Dataset.from_tensor_slices((x5_4, x5_4))
x5_4 = x5_4.batch(10)

x5_3.fit(x5_4, epochs=1, steps_per_epoch=10)

(x5_3.predict(x5_4)**2).mean()

(x5_4**2).mean()
