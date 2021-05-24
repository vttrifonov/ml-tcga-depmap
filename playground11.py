import depmap_gdc_fit as gdf
import matplotlib.pyplot as plt
from common.dir import cached_property, Dir
from common.defs import lazy_property
from types import SimpleNamespace
import types
import dask.array as daa
import zarr
from pathlib import Path
import xarray as xa
import pickle
import numpy as np

m = gdf.merge()._merge

def _zarr_data(path, data):
    if not path.exists():
        data().astype('float16').rechunk((1000, 1000)).to_zarr(str(path))
    return daa.from_zarr(zarr.open(str(path)).astype('float32'))

def _pickle_data(path, data):
    if path.exists():
        with path.open('rb') as file:
            data = pickle.load(file)
    else:
        data = data()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as file:
            pickle.dump(data, file)
    return data

class Mat:
    @lazy_property
    def rows(self):
        return _pickle_data(
            self.storage / 'rows',
            lambda: self._data.drop_dims('cols')
        )

    @lazy_property
    def cols(self):
        return _pickle_data(
            self.storage / 'cols',
            lambda: self._data.drop_dims('rows')
        )

    @lazy_property
    def data(self):
        return _zarr_data(
            self.storage/'data.zarr',
            lambda: self._data.data.data
        )

    @property
    def xarray(self):
        return xa.merge([
            self.rows,
            self.cols,
        ]).assign(
            data=(('rows', 'cols'), self.data)
        )

    class SVD:
        def __init__(self, mat):
            self.mat = mat

        @lazy_property
        def _svd(self):
            return gdf.SVD.from_data(self.mat.data)

        @lazy_property
        def u(self):
            return _zarr_data(
                self.storage / 'u.zarr',
                lambda: self._svd.u
            )

        @lazy_property
        def s(self):
            return _zarr_data(
                self.storage / 's.pickle',
                lambda: self._svd.s.persist()
            )

        @lazy_property
        def v(self):
            return _zarr_data(
                self.storage / 'v.zarr',
                lambda: self._svd.v
            )

        @lazy_property
        def pc(self):
            return xa.Dataset({'pc': np.arange(len(self.s))})

        @property
        def rows(self):
            return self.mat.rows

        @property
        def cols(self):
            return self.mat.cols

        @lazy_property
        def xarray(self):
            return xa.merge([
                self.rows,
                self.cols,
                self.pc
            ]).assign(
                u=(('rows', 'pc'), self.u),
                s=('pc', self.s),
                v=(('cols', 'pc'), self.v),
            )

    @lazy_property
    def svd(self):
        svd = self.SVD(self)
        svd.storage = self.storage/'svd'
        return svd

x1 = Mat()
x1._data = m.dm_expr
x1.storage = Path('tmp')
x1.xarray
x1.svd.xarray