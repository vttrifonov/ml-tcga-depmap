import pandas as pd
from common.defs import lazy_property
from common.dir import Dir, cached_property
from gdc.cnv import cnv as _cnv
from helpers import slice_iter, config
import numpy as np
import zarr
from joblib import Parallel, delayed
import numcodecs as nc
from pathlib import Path
import dask.array as daa
import xarray as xa
import ensembl.sql.ensembl as ensembl

class CNV:
    @lazy_property
    def storage(self):
        return Dir(config.cache/'gdc/cnv')

    @lazy_property
    @cached_property(type=Dir.pickle)
    def cases(self):
        m = _cnv.manifest
        m = m[m.project_id.str.match('TCGA')]
        m = m[['project_id', 'id', 'case_id']]
        m = m.reset_index(drop=True)
        return m

    def data(self, file):
        return _cnv.file_data(file)

    @lazy_property
    def rows(self):
        return self.cases.\
            set_index('id').\
            sort_values(['project_id'])

    @lazy_property
    @cached_property(type=Dir.pickle)
    def cols(self):
        rows = self.rows.index

        def _colnames(_range):
            colnames = set()
            slices = slice_iter(_range.start, _range.stop, 100)
            for _slice in slices:
                data = (self.data(file).gene_id for file in rows[_slice])
                data = pd.concat(data)
                colnames.update(data)
            colnames = pd.Series(list(colnames))
            return colnames

        ranges = slice_iter(0, len(rows), 1000)
        ranges = (delayed(_colnames)(range) for range in ranges)
        genes = Parallel(n_jobs=10, verbose=10)(ranges)
        genes = pd.concat(genes).drop_duplicates()
        genes = genes.sort_values().reset_index(drop=True)
        return genes

    @lazy_property
    def col_gene(self):
        result = pd.DataFrame()
        result['col'] = self.cols
        result['gene_id'] = result.col. \
            str.replace('\\..*$', '', regex=True). \
            str.replace('ENSGR', 'ENSG0', regex=True)
        return result

    @lazy_property
    @cached_property(type=Dir.pickle)
    def col_transcript(self):
        col_gene = self.col_gene
        gene_transcript = ensembl.query(ensembl.sql['gene_transcript'], 'homo_sapiens_core_104_38')
        col_transcript = col_gene.set_index('gene_id').\
            join(gene_transcript.set_index('gene_id'), how='inner')
        col_transcript = col_transcript.reset_index(drop=True).drop_duplicates()
        return col_transcript

    @lazy_property
    @cached_property(type=Dir.pickle)
    def col_go(self):
        col_transcript = self.col_transcript
        transcript_go = ensembl.query(ensembl.sql['transcript_go'], 'homo_sapiens_core_104_38')
        col_go = col_transcript.set_index('transcript_id').\
            join(transcript_go.set_index('stable_id'), how='inner')
        col_go = col_go.reset_index(drop=True).drop_duplicates()
        return col_go

    @lazy_property
    @cached_property(type=Dir.pickle)
    def col_entrez(self):
        col_gene = self.col_gene
        gene_entrez = ensembl.query(ensembl.sql['gene_entrez'], 'homo_sapiens_core_104_38')
        col_entrez = col_gene.set_index('gene_id').\
            join(gene_entrez.set_index('stable_id'), how='inner')
        col_entrez = col_entrez.reset_index(drop=True).drop_duplicates()
        return col_entrez

    @lazy_property
    def zarr(self):
        cols = self.cols
        cols = pd.Series(range(cols.shape[0]), index=cols)

        rows = self.rows.index

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
                    data = (self.data(file).set_index('gene_id').copy_number for file in rows[_slice])
                    data = (cols.align(data, join='left')[1] for data in data)
                    data = pd.concat(data)
                    data = np.array(data, dtype='float16')
                    data = data.reshape((_slice.stop-_slice.start, -1))
                    mat[_slice, :] = data

            ranges = slice_iter(0, len(rows), 1000)
            ranges = (delayed(_load_data)(range) for range in ranges)
            Parallel(n_jobs=10, verbose=10)(ranges)

        return zarr.open(str(path), mode='r')

    @lazy_property
    def xarray(self):
        result = xa.Dataset()
        result['data'] = (['rows', 'cols'], daa.from_zarr(self.zarr))
        result = result.merge(self.rows.rename_axis('rows'))
        result['cols'] = np.array(self.cols)
        return result

cnv = CNV()

