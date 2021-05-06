from pathlib import Path
from common.dir import Dir
from helpers import config
import ae_expr as ae

config.exec()

storage = Path('output/playground2')
storage.mkdir(parents=True, exist_ok=True)

model1 = ae.model1()
model1.data =  ae.data1()
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

class _data:
    @property
    def data1(self):
        mat = xa.Dataset()
        mat['data'] = (('rows', 'cols'), daa.from_array(np.random.randn(1000, 500)))

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
        return np.cross(np.arange(2), np.arange(3))

        m = self.num_cols
        ij = np.array(list(it.product(np.arange(10), np.arange(m))))
        ij = 