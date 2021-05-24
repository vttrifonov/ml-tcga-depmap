import pandas as pd
import xarray as xa
import numpy as np
from common.defs import lazy_property
from depmap.depmap import public_21q1 as release
from common.dir import cached_property, Dir
from helpers import config
from pathlib import Path
import zarr
import numcodecs as nc
import dask.array as daa

config.exec()

class CNV:
    @property
    def release(self):
        return release

    @lazy_property
    def storage(self):
        return Dir(config.cache).child('depmap').child('cnv')

    @lazy_property
    def _data(self):
        return release.gene_cnv

    @lazy_property
    @cached_property(type=Dir.pickle)
    def rows(self):
        return self._data.iloc[:, 0].rename('rows')

    @lazy_property
    @cached_property(type=Dir.pickle)
    def cols(self):
        cols = pd.Series(self._data.columns[1:]).to_frame('cols')
        cols['symbol'] = cols.cols.str.replace(' .*$', '', regex=True)
        cols['entrez'] = cols.cols.str.replace('^.*\(|\)$', '', regex=True).astype(int)
        return cols

    @lazy_property
    def mat(self):
        path = Path(self.storage.path) / 'mat.zarr'
        if not path.exists():
            mat = np.array(self._data.iloc[:, 1:]).astype('float16')
            z = zarr.open(
                str(path), mode='w',
                shape=mat.shape, dtype=mat.dtype,
                chunks=(1000, 1000),
                compressor=nc.Blosc(cname='zstd', clevel=3)
            )
            z[:,:] = mat
        return zarr.open(str(path), mode='r')

    @lazy_property
    def data(self):
        rows = self.rows
        cols = self.cols
        mat = self.mat

        data = xa.Dataset()
        data['rows'] = ('rows', rows)
        data = data.merge(cols.set_index('cols').to_xarray())
        data['mat'] = (('rows', 'cols'), daa.from_zarr(mat))

        data = data.merge(
            self.release.samples.rename(columns={'DepMap_ID': 'rows'}).set_index('rows').to_xarray(),
            join='inner'
        )

        return data

cnv = CNV()