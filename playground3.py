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

storage = Path('output/playground3')
#shutil.rmtree(storage)
storage.mkdir(parents=True, exist_ok=True)

model = ae.model2()
model.data =  ae.data2()
model.kwargs = {'cp_callback': {'save_freq': 121, 'verbose': 1}}
model.data.storage = Dir(storage/'data')
model.storage = Path(model.data.storage.path)/'model3'
model.fit(epochs=100, steps_per_epoch=12)

model.data.data1.go.query('n>=5').sort_values('n')


