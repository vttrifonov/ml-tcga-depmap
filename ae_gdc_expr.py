import numpy as np
import pandas as pd
import dask_ml.preprocessing as dmlp
import tensorflow as tf
import tensorflow.keras as tfk
from common.defs import lazy_property
from common.dir import Dir, cached_property
from types import SimpleNamespace
import dask.array as daa
from gdc_expr import expr
from helpers import chunk_iter
from ae import Sparse1
import datetime
import json

class _data:
    @lazy_property
    def data1(self):
        mat2 = expr.mat2

        mat = mat2.xarray
        mat['mean'] = mat.data.mean(axis=0).compute()
        mat = mat.sel(cols=mat['mean']>(-7))

        cols = pd.DataFrame({'col': mat.cols.values})
        cols['j'] = range(cols.shape[0])

        col_go = mat2.col_go[['col', 'display_label']].rename(columns={'display_label': 'go'}).drop_duplicates()
        col_go = cols.set_index('col').join(col_go.set_index('col'), how='inner').reset_index()
        go = col_go.value_counts('go').rename('n').reset_index()

        return SimpleNamespace(
            mat = mat.data.data,
            cols = cols,
            col_go = col_go,
            go = go
        )

    def _split(self, ratio):
        data1 = self.data1.mat
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
        split = self.split
        data1 = self.data1
        data1.train = daa.vstack([data1.mat[chunk[0], :] for chunk in split])
        data1.test = daa.vstack([data1.mat[chunk[1], :] for chunk in split])
        scaler = dmlp.StandardScaler().fit(data1.train)
        data1.train = scaler.transform(data1.train)
        data1.test = scaler.transform(data1.test)
        return data1

    @property
    def num_cols(self):
        return self.data2.train.shape[1]

    @lazy_property
    def ij(self):
        data2 = self.data2
        go = data2.go.query('(n >= 10) & (n <= 700)').copy()
        go['i'] = range(go.shape[0])
        ij = data2.col_go.set_index('go').join(go.set_index('go').i, how='inner')[['i', 'j']]
        ij = ij.reset_index(drop=True).sort_values(['i', 'j'])
        return ij

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
        return model

    def fit(self, **kwargs):
        model = self.model
        history = model.fit(
            *self.data.data.Xy,
            **{
                'callbacks': [self.cp_callback],
                **self.kwargs.get('fit', {}),
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
    batch_size = 800

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
                batch_size = self.batch_size
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

class model3(_fit):
    def build(self):
        ij = self.data.ij.copy()
        ij['j'] = np.random.permutation(ij.j)
        model = tfk.Sequential([
            tfk.layers.InputLayer((self.data.num_cols,)),
            Sparse1(np.array(ij), (max(ij.i)+1, self.data.num_cols)),
            tfk.layers.Dense(self.data.num_cols)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model



