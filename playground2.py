import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common.defs import pipe, lapply, lfilter
import seaborn as sb
import more_itertools as mit
import functools as ft
from expr import expr
import zarr
from joblib import Parallel, delayed
from pathlib import Path
from common.defs import lazy_property
from common.dir import cached_property, Dir
import itertools as it
import numcodecs as nc
from pathlib import Path

def slice_iter(b, e, c):
    if b % c != 0:
        b1 = c*np.ceil(b/c)
        yield slice(b, b1)
        b = b1
    for i in range(b, e, c):
        yield slice(i, min(i+c, e))

class Mat:
    def __init__(self, expr):
        self.expr = expr

    @lazy_property
    def storage(self):
        return self.expr.storage.child('mat')

    @lazy_property
    def cols(self):
        return self.expr.files.\
            set_index('id').\
            sort_values(['project_id', 'sample_type', 'is_normal'])

    @lazy_property
    @cached_property(type=Dir.pickle)
    def rows(self):
        def _rownames(files):
            rownames = set()
            files = mit.chunked(files, 100)
            for files1 in files:
                data = (self.expr.data(file).Ensembl_Id for file in files1)
                data = pd.concat(data)
                rownames.update(data)
            rownames = pd.Series(list(rownames))
            return rownames

        genes = self.cols.index
        genes = mit.chunked(genes, 1000)
        genes = (delayed(_rownames)(files) for files in genes)
        genes = Parallel(n_jobs=10, verbose=10)(genes)
        genes = pd.concat(genes).drop_duplicates()
        genes = genes.sort_values().reset_index(drop=True)
        return genes

    @lazy_property
    def zarr(self):
        rows = self.rows
        rows = pd.Series(range(rows.shape[0]), index=rows)

        cols = self.cols.index

        path = Path(self.storage.child('data.zarr').path)
        if not path.exists():
            mat = zarr.open(
                str(path), mode='w',
                shape=(len(rows), len(cols)), dtype='float16',
                chunks=(1000, 1000),
                compressor=nc.Blosc(cname='zstd', clevel=3)
            )
            def _load_data(_range):
                slices = slice_iter(_range.start, _range.stop, 100)
                for _slice in slices:
                    print(_slice.start)
                    data = (np.log2(self.expr.data(file).set_index('Ensembl_Id').value+1e-3) for file in cols[_slice])
                    data = (rows.align(data, join='left')[1] for data in data)
                    data = pd.concat(data)
                    data = np.array(data, dtype='float16')
                    data = data.reshape((_slice.stop-_slice.start, -1))
                    mat[:, _slice] = data.T

            ranges = slice_iter(0, len(cols), 1000)
            ranges = (delayed(_load_data)(range) for range in ranges)
            Parallel(n_jobs=10, verbose=10)(ranges)

        return zarr.open(str(path), mode='r')

self = Mat(expr)

self.rows

self.zarr

