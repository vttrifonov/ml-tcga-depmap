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

class Sparse(tfk.layers.Layer):
    def __init__(self, ij, dense_shape):
        super().__init__()
        v = self.add_weight(
            name='sparse_kernel',
            shape=(ij.shape[0],),
            trainable=True
        )
        self.sparse_kernel = tf.SparseTensor(
            indices=ij,
            values=v,
            dense_shape=dense_shape
        )
        self.sparse_kernel = tf.sparse.reorder(self.sparse_kernel)

    def call(self, inputs):
        outputs = tf.transpose(inputs)
        outputs = tf.sparse.sparse_dense_matmul(self.sparse_kernel, outputs)
        outputs = tf.transpose(outputs)
        return outputs

class Sparse1(tfk.layers.Layer):
    def __init__(self, ij, dense_shape):
        super().__init__()
        self.sparse_kernel = self.add_weight(
            name='sparse_kernel',
            shape=dense_shape,
            trainable=True
        )

    def call(self, inputs):
        outputs = tf.transpose(inputs)
        outputs = tf.linalg.matmul(self.sparse_kernel, outputs)
        outputs = tf.transpose(outputs)
        return outputs

class Sparse2(tfk.layers.Layer):
    def __init__(self, ij, dense_shape):
        super().__init__()
        self.sparse_kernel = self.add_weight(
            name='sparse_kernel',
            shape=(ij.shape[0],1),
            trainable=True
        )
        self.dense_shape = dense_shape
        u = np.unique(ij[:,0], return_index=True)
        u = u[1][1:]
        self.kj = (
            tf.ragged.constant(np.split(np.arange(ij.shape[0]), u)),
            tf.ragged.constant(np.split(ij[:,1], u))
        )

    def call(self, inputs):
        def _mult(k, j):
            input = tf.gather(inputs, j, axis=1)
            kernel = tf.gather(self.sparse_kernel, k, axis=0)
            result = tf.tensordot(input, kernel, 1)
            return result

        outputs = tf.map_fn(
            lambda kj: _mult(*kj),
            self.kj,
            fn_output_signature=tf.float32
        )
        return outputs

class Sparse3(tfk.layers.Layer):
    def __init__(self, ij, dense_shape):
        super().__init__()
        self.sparse_kernel = self.add_weight(
            name='sparse_kernel',
            shape=(ij.shape[0],),
            trainable=True
        )
        self.dense_shape = dense_shape
        self.ids = tf.SparseTensor(
            indices = ij,
            values = np.arange(ij.shape[0]),
            dense_shape=dense_shape
        )
        self.j = ij[:,1]


    def call(self, inputs):
        def _mult(input):
            weights = tf.SparseTensor(
                indices=self.ids.indices,
                values=tf.gather(input, self.j, axis=0),
                dense_shape=self.ids.dense_shape
            )
            outputs = tf.nn.embedding_lookup_sparse(
                self.sparse_kernel,
                sp_ids=self.ids,
                sp_weights=weights,
                combiner='sum'
            )
            print(outputs.shape)
            return outputs

        outputs = tf.map_fn(_mult, inputs)
        return outputs

x4_1 = x1.sel(cols=x1['mean']>(-7))
x4_4 = dmlp.StandardScaler().fit_transform(x4_1.data.data)

x4_3 = self.col_go[['col', 'display_label']].drop_duplicates().set_index('col')
x4_2 = pd.DataFrame({'col': x4_1.cols.values}).reset_index().rename(columns={'index': 'j'})
x4_2 = x4_2.set_index('col').join(x4_3, how='inner').set_index('display_label')
x4_3 = x4_2.index.value_counts()
x4_3 = x4_3[(x4_3>=10) & (x4_3<=700)]
x4_3 = x4_3[x4_3.index.isin(['GO:0003149', 'GO:0033089'])]
x4_3 = x4_3.reset_index().reset_index().set_index('index').rename(columns={'level_0': 'i'})[['i']]
x4_2 = x4_2.join(x4_3, how='inner')
x4_2 = x4_2[['i', 'j']].reset_index(drop=True).sort_values(['i', 'j'])

x6_1 = Sparse2(np.array(x4_2), (x4_2.i.max()+1, x4_4.shape[1]))
inputs = tfk.layers.Input(x4_4.shape[1])
self = x6_1

tf.map_fn(lambda x: (print(x[0].shape, x[1].shape), x)[1], x6_1.kj)


x5_3 = tfk.Sequential([
    tfk.layers.InputLayer((x4_4.shape[1],)),
    Sparse(np.array(x4_2), (x4_2.i.max()+1, x4_4.shape[1])),
    tfk.layers.Dense(x4_4.shape[1])
])
x5_3.compile(optimizer='adam', loss='mean_squared_error')
x5_3.summary()

def chunk_iter(chunks):
    chunks = np.cumsum(chunks)
    chunks = zip(np.hstack([0, chunks[:-1]]), chunks)
    for x in chunks:
        yield slice(x[0], x[1])

def x5_4_1():
    for chunk in chunk_iter(x4_4.chunks[0]):
        x = x4_4[chunk,:].compute()
        for i in range(x.shape[0]):
            yield x[i,:]

x5_4 = tf.data.Dataset.from_generator(
    lambda: ((row, row) for row in x5_4_1()),
    output_types=(tf.float32, tf.float32),
    output_shapes=((x4_4.shape[1],), (x4_4.shape[1],))
)
x5_4 = x5_4.batch(1000).repeat()

x5_3.trainable_weights[0]
x5_3.fit(x5_4, epochs=1, steps_per_epoch=3)

(x5_3.predict(x5_4)**2).mean()

(x5_4**2).mean()

