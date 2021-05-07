from pathlib import Path
from common.dir import Dir
from helpers import config
import ae_expr as ae

config.exec()

storage = Path('output/playground2')
storage.mkdir(parents=True, exist_ok=True)

model1 = ae.model1()
model1.data =  ae.data2()
model1.data.storage = Dir(storage/'data')
model1.storage = Path(model1.data.storage.path)/'model1'
model1.fit(epochs=50)

model2 = ae.model2()
model2.data =  ae.data1()
model2.data.storage = Dir(storage/'data')
model2.storage = Path(model2.data.storage.path)/'model2'
model2.fit(epochs=100)


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
from helpers import chunk_iter
from ae import Sparse1
import xarray as xa
import itertools as it
import shutil
import matplotlib.pyplot as plt

class _data(ae.data1):
    @property
    def data1(self):
        mat = xa.Dataset()
        mat['data'] = (('rows', 'cols'), daa.from_array(np.random.randn(1000, 500)))
        return mat

    def _split(self, ratio):
        data1 = self.data1.data.data
        chunks = data1.chunks[0]

        perm = (
            np.random.permutation(np.arange(chunk.start, chunk.stop))
            for chunk in chunk_iter(chunks)
        )

        perm = (
            (x[:int(ratio*len(x))], x[int(ratio*len(x)):])
            for x in perm
        )

        perm = list(perm)
        return perm

    @lazy_property
    def split(self):
        return self._split(0.8)

    @lazy_property
    def data2(self):
        x1 = self.data1

        split = self.split
        x2 = SimpleNamespace()
        x2.train = daa.vstack([
            x1.data.data[chunk[0],:]
            for chunk in split
        ])
        x2.test = daa.vstack([
            x1.data.data[chunk[1],:]
            for chunk in split
        ])

        x3 = dmlp.StandardScaler().fit(x2.train)
        x2.train = x3.transform(x2.train)
        x2.test = x3.transform(x2.test)
        return x2

    @property
    def num_cols(self):
        return self.data2.train.shape[1]

    @lazy_property
    def ij(self):
        m = self.num_cols
        ij = np.array(list(it.product(np.arange(200), np.arange(m))))
        ij = pd.DataFrame(ij).rename(columns={0: 'i', 1: 'j'})
        return ij

    @lazy_property
    def data(self):
        data2 = self.data2
        data2 = SimpleNamespace(
            train=data2.train.compute(),
            test=data2.test.compute()
        )
        data4 = SimpleNamespace(
            Xy=[data2.train, data2.train],
            kwargs=dict(
                validation_data=(data2.test, data2.test),
                batch_size=50
            )
        )
        return data4

model1 = ae.model1()
model1.data = _data()
model1.data.storage = Dir(storage/'_data')
model1.storage = Path(model1.data.storage.path)/'model1'
shutil.rmtree(model1.data.storage.path, ignore_errors=True)
model1.data.storage.exists
model1.fit(epochs=1)

model2 = ae.model2()
model2.data = _data()
model2.data.storage = Dir(storage/'_data')
model2.storage = Path(model2.data.storage.path)/'model2'
model2.fit(epochs=1)

x1_1 = model1.data.data2.train
x1_2 = model1.data.data2.test
x2 = dmld.PCA(n_components=max(model1.data.ij.i)+1).fit(x1_1)
x3_1 = x2.inverse_transform(x2.transform(x1_1))
x3_2 = x2.inverse_transform(x2.transform(x1_2))
x4_1 = model2.model.predict(x1_1)
x4_2 = model2.model.predict(x1_2)

print(((x1_1-x3_1)**2).mean().compute())
print(((x1_2-x3_2)**2).mean().compute())

print(((x1_1-x4_1)**2).mean().compute())
print(((x1_2-x4_2)**2).mean().compute())

print(((x3_1-x4_1)**2).mean().compute())
print(((x3_2-x4_2)**2).mean().compute())




