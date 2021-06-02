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
import ncbi.sql.ncbi as ncbi
import ucsc_gb.sql as ucsc

config.exec()

def _loc_dummy(x):
    import sparse
    import re

    loc = x.to_series()
    loc = loc.str.replace(' and ', '|')
    loc1 = [np.array(x.split('|')) for x in loc]
    loc2 = [np.repeat(loc.index[i], len(x)) for i, x in enumerate(loc1)]
    loc = pd.Series(
        np.hstack(loc1),
        index=pd.Index(np.hstack(loc2), name=loc.index.name),
        name=loc.name
    )
    loc = loc.reset_index().drop_duplicates().set_index(loc.index.name).squeeze()

    def _split(s):
        s = s.split('-')
        if len(s)==1:
            return s
        chr = re.sub('[pq].*$', '', s[0])
        s[1] = chr + s[1]
        return s

    loc1 = [np.array(_split(x)) for x in loc]
    loc2 = [np.repeat(loc.index[i], len(x)) for i, x in enumerate(loc1)]
    loc = pd.Series(
        np.hstack(loc1),
        index=pd.Index(np.hstack(loc2), name=loc.index.name),
        name=loc.name
    )
    loc = loc.reset_index().drop_duplicates().set_index(loc.index.name).squeeze()

    rows = pd.Series(range(len(x.cols)), index=x.cols)
    cols = loc.drop_duplicates()
    cols = pd.Series(range(len(cols)), index=cols)
    data = sparse.COO([rows[loc.index], cols[loc]], 1, shape=(len(rows), len(cols)))
    data = data.todense()
    data = daa.from_array(data, chunks=(-1, -1))
    loc = xa.DataArray(
        data,
        dims=('cols', x.name + '_cols'),
        coords={
            'cols': rows.index.to_numpy(),
            x.name + '_cols': cols.index.to_numpy()
        },
        name=x.name
    )
    return loc

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
    @cached_property(type=Dir.pickle)
    def col_map_location(self):
        cols = self.cols
        map_location = ncbi.query(ncbi.sql['map_location'], 'homo_sapiens')
        map_location = cols.set_index('entrez').\
            join(map_location.set_index('entrez'), how='inner')
        map_location = map_location.reset_index(drop=True)[['cols', 'map_location']].drop_duplicates()
        return map_location

    @lazy_property
    @cached_property(type=Dir.pickle)
    def col_tx(self):
        cols = self.cols
        tx = ucsc.query(ucsc.sql['refseq_tx'], 'hg38')
        tx = tx[['name2', 'chrom', 'strand', 'txStart', 'txEnd']]
        tx = tx.groupby(['name2', 'chrom', 'strand']).agg(
            txStart_min=('txStart', 'min'),
            txStart_max=('txStart', 'max'),
            txEnd_min=('txEnd', 'min'),
            txEnd_max=('txEnd', 'max')
        )
        tx = tx.reset_index()
        tx['txStart'] = np.where(tx.strand=='+', tx.txStart_min, tx.txStart_max)
        tx['txEnd'] = np.where(tx.strand == '+', tx.txEnd_max, tx.txEnd_min)
        tx = tx.groupby('name2').filter(lambda x: x.shape[0]==1)
        tx = cols.set_index('symbol').\
            join(tx.set_index('name2'), how='inner')
        tx = tx.reset_index(drop=True)[['cols', 'chrom', 'strand', 'txStart', 'txEnd']].drop_duplicates()
        return tx

    @property
    def row_annot(self):
        return self.release.samples.rename(columns={'DepMap_ID': 'rows'})

    @lazy_property
    def mat1(self):
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
        mat = mat.merge(self.col_tx.set_index('cols'), join='inner')
        mat = mat.sel(cols=np.isnan(mat.data).sum(axis=0)==0)
        data = mat.data.data
        data = data.rechunk(-1, 1000).astype('float32')
        data = daa.log2(data+0.1)
        mat['data'] = (('rows', 'cols'), data)

        mat = mat.rename({'map_location': 'cyto'})
        mat['arm'] = mat.cyto.str.replace('^([^pq]*[pq]).*$', r'\1', regex=True)
        mat = mat.sel(cols=mat.arm!='-')

        mat['cyto_dummy'] = _loc_dummy(mat.cyto)
        return mat

cnv = CNV()