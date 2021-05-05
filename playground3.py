import importlib
import numpy as np
import pandas as pd
from pathlib import Path
import dask_ml.preprocessing as dmlp
import tensorflow as tf
import tensorflow.keras as tfk
from datetime import datetime

import expr
import helpers
import ae
importlib.reload(expr)
importlib.reload(helpers)
importlib.reload(ae)
from expr import expr
from helpers import chunk_iter, config
from ae import Sparse1

config.exec()

self = expr.mat2

x1 = self.xarray
x1['var'] = x1.data.var(axis=0).compute()
x1['mean'] = x1.data.mean(axis=0).compute()

x4_1 = x1.sel(cols=x1['mean']>(-7))
x4_4 = dmlp.StandardScaler().fit_transform(x4_1.data.data)

x4_3 = self.col_go[['col', 'display_label']].drop_duplicates().set_index('col')
x4_2 = pd.DataFrame({'col': x4_1.cols.values}).reset_index().rename(columns={'index': 'j'})
x4_2 = x4_2.set_index('col').join(x4_3, how='inner').set_index('display_label')
x4_3 = x4_2.index.value_counts()
x4_3 = x4_3[(x4_3>=10) & (x4_3<=700)]
#x4_3 = x4_3[x4_3.index.isin(['GO:0048471','GO:0003149', 'GO:0033089'])]
#x4_3 = x4_3[x4_3.index.isin(['GO:0003149', 'GO:0033089'])]
x4_3 = x4_3[x4_3.index.isin(['GO:0048471'])]
x4_3 = x4_3.reset_index().reset_index().set_index('index').rename(columns={'level_0': 'i'})[['i']]
x4_2 = x4_2.join(x4_3, how='inner')
x4_2 = x4_2[['i', 'j']].reset_index(drop=True).sort_values(['i', 'j'])

def _x4_5():
    for chunk in chunk_iter(x4_4.chunks[0]):
        x = x4_4[chunk,:].compute()
        for i in range(x.shape[0]):
            yield x[i,:]
x4_5 = tf.data.Dataset.from_generator(_x4_5, output_types=x4_4.dtype, output_shapes=(x4_4.shape[1],)).\
    map(lambda row: (row, row)).\
    batch(1000).repeat()


x5_3 = tfk.Sequential([
    tfk.layers.InputLayer((x4_4.shape[1],)),
    Sparse1(np.array(x4_2), (max(x4_2.i)+1, x4_4.shape[1])),
    #tfk.layers.Dense(x4_2.shape[0], use_bias=False),
    tfk.layers.Dense(x4_4.shape[1])
])
x5_3.compile(optimizer='adam', loss='mse')
x5_3.summary()

x5_3.fit(x4_5, epochs=2, steps_per_epoch=12)

models = Path(self.storage.path)/'models'

x5_3.save(models/datetime.now().strftime("%Y%m%d%H%M%S"))
