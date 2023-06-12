import pandas as pd
import xarray as xa
import numpy as np
from .common.defs import lazy_property
from .depmap.depmap import public_21q1 as release
from .common.dir import cached_property, Dir
from .helpers import config
from pathlib import Path
import zarr
import numcodecs as nc
import dask.array as daa
from .ncbi.sql import ncbi

config.exec()

class CRISPR:
    @property
    def release(self):
        return release

    @lazy_property
    def storage(self):
        return Dir(config.cache).child('depmap').child('crispr')

    @lazy_property
    def _crispr(self):
        return release.crispr_effect

    @lazy_property
    @cached_property(type=Dir.pickle)
    def rows(self):
        return self._crispr.iloc[:, 0].rename('rows')

    @lazy_property
    @cached_property(type=Dir.pickle)
    def cols(self):
        cols = pd.Series(self._crispr.columns[1:]).to_frame('cols')
        cols['symbol'] = cols.cols.str.replace(' .*$', '', regex=True)
        cols['entrez'] = cols.cols.str.replace('^.*\(|\)$', '', regex=True).astype(int)
        return cols

    @lazy_property
    @cached_property(type=Dir.pickle)
    def col_map_location(self):
        cols = self.cols
        map_location = ncbi.query(ncbi.sql['map_location'], 'homo_sapiens')
        map_location = cols.set_index('entrez').\
            join(map_location.set_index('entrez'), how='inner')
        map_location = map_location.reset_index(drop=True)[['cols', 'map_location']].drop_duplicates()
        return map_location

    @property
    def row_annot(self):
        return self.release.samples.rename(columns={'DepMap_ID': 'rows'})

    @lazy_property
    def mat1(self):
        path = Path(self.storage.path) / 'mat.zarr'
        if not path.exists():
            mat = np.array(self._crispr.iloc[:, 1:]).astype('float16')
            z = zarr.open(
                str(path), mode='w',
                shape=mat.shape, dtype=mat.dtype,
                chunks=(1000, 1000),
                compressor=nc.Blosc(cname='zstd', clevel=3)
            )
            z[:,:] = mat
        return zarr.open(str(path), mode='r')

    @lazy_property
    def mat2(self):
        rows = self.rows
        cols = self.cols
        mat = self.mat1

        data = xa.Dataset()
        data['rows'] = ('rows', rows)
        data = data.merge(cols.set_index('cols').to_xarray())
        data['data'] = (('rows', 'cols'), daa.from_zarr(mat))

        return data

    @lazy_property
    def mat3(self):
        mat = self.mat2.copy()
        mat = mat.merge(self.row_annot.set_index('rows'), join='inner')
        mat = mat.merge(self.col_map_location.set_index('cols'), join='inner')
        mat = mat.sel(cols=np.isnan(mat.data).sum(axis=0)==0)
        mat['data'] = (('rows', 'cols'), mat.data.data.rechunk(-1, 1000))
        return mat

crispr = CRISPR()
