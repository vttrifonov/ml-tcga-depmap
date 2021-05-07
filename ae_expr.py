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
import pickle
import datetime
import json

class _data:
    @property
    def data1(self):
        return expr.mat2

    def _split(self, ratio):
        data1 = self.data1.xarray.data.data
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
    @cached_property(type=Dir.pickle)
    def split(self):
        return self._split(0.8)

    @lazy_property
    def data2(self):
        x1 = self.data1.xarray
        x1['mean'] = x1.data.mean(axis=0).compute()
        x1 = x1.sel(cols=x1['mean']>(-7))

        split = self.split
        x2 = SimpleNamespace()
        x2.cols = x1.cols.values
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
        x4_1 = self.data2

        x4_3 = self.data1.col_go[['col', 'display_label']].rename(columns={'display_label': 'go'}).drop_duplicates()
        x4_2 = pd.DataFrame({'col': x4_1.cols})
        x4_2['j'] = range(x4_2.shape[0])
        x4_2 = x4_2.set_index('col').join(x4_3.set_index('col'), how='inner').set_index('go')
        x4_3 = x4_2.index.value_counts()
        x4_3 = x4_3[(x4_3 >= 10) & (x4_3 <= 700)]
        x4_3 = x4_3.to_frame()
        x4_3['i'] = range(x4_3.shape[0])
        x4_2 = x4_2.join(x4_3.i, how='inner')
        x4_2 = x4_2[['i', 'j']].reset_index(drop=True).sort_values(['i', 'j'])
        return x4_2

class _fit:
    kwargs = {}

    @property
    def cp_callback(self):
        callback = tfk.callbacks.ModelCheckpoint(
            filepath=self.storage,
            **self.kwargs.get('cp_callback', {})
        )
        return callback

    @lazy_property
    def model(self):
        if self.storage.exists():
            print(f'loading model from {self.storage}')
            model = tfk.models.load_model(self.storage)
            return model
        model = self.build()
        print(f'saving model to {self.storage}')
        model.save(self.storage)
        return model

    def fit(self, **kwargs):
        model = self.model
        history = model.fit(
            *self.data.data.Xy,
            **{
                **self.kwargs.get('fit', {}),
                'callbacks': [self.cp_callback],
                **self.data.data.kwargs,
                **kwargs
            }
        )
        history_file = self.storage / 'history'
        history_file.mkdir(parents=True, exist_ok=True)
        history_file = history_file/datetime.datetime.now().strftime('%Y%m%d%H%M%S.json')
        print(f'saving history to {history_file}')
        with open(history_file, 'wt') as file:
            json.dump(history.history, file)
        print(f'saving model to {self.storage}')
        model.save(self.storage)

class data1(_data):
    @lazy_property
    def data(self):
        def _ds(data):
            def _gen():
                for chunk in chunk_iter(data.chunks[0]):
                    x = data[chunk, :].compute()
                    for i in range(x.shape[0]):
                        yield x[i, :]

            ds = tf.data.Dataset.from_generator(
                _gen, output_types=tf.float32, output_shapes=(data.shape[1],)
            )
            ds = ds.map(lambda row: (row, row))
            return ds

        data2 = self.data2
        data3 = SimpleNamespace(
            Xy = [_ds(data2.train).batch(800).prefetch(2).repeat()],
            kwargs = dict(
                validation_data = _ds(data2.test).batch(200).prefetch(2),
                validation_steps = 12,
                steps_per_epoch = 12
            )
        )
        return data3

class data2(_data):
    @lazy_property
    def data(self):
        data2 = self.data2
        data2 = SimpleNamespace(
            train = data2.train.compute(),
            test = data2.test.compute()
        )
        data4 = SimpleNamespace(
            Xy = [data2.train, data2.train],
            kwargs = dict(
                validation_data = (data2.test, data2.test),
                batch_size = 800
            )
        )
        return data4

class model1(_fit):
    def build(self):
        model = tfk.Sequential([
            tfk.layers.InputLayer((self.data.num_cols,)),
            Sparse1(np.array(self.data.ij), (max(self.data.ij.i)+1, self.data.num_cols)),
            tfk.layers.Dense(self.data.num_cols)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

class model2(_fit):
    def build(self):
        model = tfk.Sequential([
            tfk.layers.InputLayer((self.data.num_cols,)),
            tfk.layers.Dense(max(self.data.ij.i)+1, use_bias=False),
            tfk.layers.Dense(self.data.num_cols)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

