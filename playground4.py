from pathlib import Path
import shutil
import importlib
import numpy as np
import pandas as pd
import dask_ml.preprocessing as dmlp
import dask_ml.decomposition as dmld
import tensorflow as tf
import tensorflow.keras as tfk
from common.defs import lazy_property
from common.dir import Dir, cached_property
from types import SimpleNamespace
import dask.array as daa
from expr import expr
from helpers import chunk_iter, config
import xarray as xa
import itertools as it
import matplotlib.pyplot as plt

import ae_expr
importlib.reload(ae_expr)
import ae_expr as ae

config.exec()

storage = Path('output/playground4')
#shutil.rmtree(storage)
storage.mkdir(parents=True, exist_ok=True)

class _data(ae.data2):
    batch_size = 100

    @property
    def data1(self):
        return SimpleNamespace(
            mat = daa.from_array(np.random.randn(1000, 3000).astype('float32'))
        )

    @lazy_property
    def split(self):
        return self._split(0.8)

    @property
    def num_cols(self):
        return self.data2.train.shape[1]

    @lazy_property
    def ij(self):
        m = self.num_cols
        ij = np.array(list(it.product(np.arange(400), np.arange(m))))
        ij = pd.DataFrame(ij).rename(columns={0: 'i', 1: 'j'})
        return ij

model = ae.model1()
model.data = _data()
model.data.storage = Dir(storage/'_data')
model.storage = Path(model.data.storage.path)/'model'
shutil.rmtree(model.data.storage.path, ignore_errors=True)
model.fit(epochs=100, callbacks=[])

import sklearn.decomposition as skld

x1_1 = model.data.data2.train.compute()
x1_2 = model.data.data2.test.compute()
x2 = np.linalg.svd(x1_1)[2][:max(model.data.ij.i)+1,:]
x3_1 = ((x1_1 @ x2.T) @ x2)
x3_2 = ((x1_2 @ x2.T) @ x2)
x4_1 = model.model.predict(x1_1)
x4_2 = model.model.predict(x1_2)

xx1 = np.linalg.svd(model.model.trainable_weights[1].numpy(), full_matrices=False)
print(((x1_1 - x3_1)**2).mean())
print(((x1_1 - (x1_1 @ xx1[2].T) @ xx1[2])**2).mean())

xx2 = xx1[2].T @ xx1[2]
xx3 = x2.components_.T @ x2.components_
plt.plot(xx2.ravel(), xx3.ravel(), '.')

pd.Series(np.round(np.log10(np.abs(xx3.ravel())))).value_counts().sort_index()

print(((x1_1-x3_1)**2).mean().compute())
print(((x1_2-x3_2)**2).mean().compute())

print(((x1_1-x4_1)**2).mean().compute())
print(((x1_2-x4_2)**2).mean().compute())

print(((x3_1-x4_1)**2).mean().compute())
print(((x3_2-x4_2)**2).mean().compute())

x5_1 = ((x1_2-x4_2)**2).mean(axis=0).compute()
plt.hist(x5_1)
plt.show()