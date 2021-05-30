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
import json
import sparse

config.exec()

def _add_locs(x):
    x = x.rename({'map_location': 'cyto'})
    map = x.cyto.to_dataframe()
    map['chr'] = map.cyto.str.replace('[pq].*$', '', regex=True)
    map['pq'] = map.cyto.str.replace('^.*([pq]).*$', r'\1', regex=True)
    map['loc'] = map.cyto.str.replace('^.*[pq]', '', regex=True)
    map['loc'] = pd.to_numeric(map['loc'], errors='coerce')
    map['loc'] = (2 * (map.pq == 'q') - 1) * map['loc']
    map['arm'] = map.chr + map.pq
    map = map.sort_values(['chr', 'loc'])
    x = x.merge(map[['chr', 'loc', 'arm']])
    x = x.sel(cols=map.index)
    x = x.sel(cols=~x['loc'].isnull())
    x.data.values = x.data.data.rechunk('auto').persist()
    return x

class CNV:
    @lazy_property
    def storage(self):
        return Dir(config.cache/'gdc/cnv')

    @lazy_property
    @cached_property(type=Dir.pickle)
    def files(self):
        m = _cnv.manifest
        m = m[m.project_id.str.match('TCGA')]
        m = m.reset_index(drop=True)
        s = (json.loads(x) for x in m.samples)
        s = (pd.DataFrame(x) for x in s)
        s = (x.query('sample_type.str.find("Normal")<0') for x in s)
        s = (x['sample_id'] for x in s)
        s = pd.concat(s)
        s = s.reset_index(drop=True)
        m['sample_id'] = s
        m = m[['project_id', 'id', 'case_id', 'sample_id']]
        return m

    def data(self, file):
        return _cnv.file_data(file)

    @lazy_property
    def rows(self):
        return self.files.\
            set_index('id').\
            sort_values(['project_id', 'case_id', 'sample_id'])

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
    def mat1(self):
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
    def mat2(self):
        result = xa.Dataset()
        result['data'] = (['rows', 'cols'], daa.from_zarr(self.mat1))
        result = result.merge(self.rows.rename_axis('rows'))
        result['cols'] = np.array(self.cols)
        return result

    @lazy_property
    def mat3_cols(self):
        x1_1 = self.col_entrez[['col', 'dbprimary_acc', 'display_label']]
        x1_1 = x1_1.drop_duplicates()
        x1_1 = x1_1.rename(columns={
            'col': 'cols',
            'dbprimary_acc': 'entrez',
            'display_label': 'symbol'
        })
        x1_1['new_cols'] = x1_1.symbol + ' (' + x1_1.entrez + ')'
        x1_1['n'] = x1_1.groupby('new_cols').new_cols.transform('size')
        x1_1 = x1_1.query('n==1 | cols.str.find("ENSGR")<0').copy()
        x1_1['n'] = x1_1.groupby('new_cols').new_cols.transform('size')
        x1_1 = x1_1.query('n==1').copy()
        del x1_1['n']
        x1_1 = x1_1.set_index('cols').to_xarray()
        return x1_1

    @lazy_property
    def mat3_rows(self):
        x5_1 = self.mat2[['rows', 'project_id','case_id', 'sample_id']]
        x5_1 = x5_1.to_dataframe().reset_index()
        x5_1 = x5_1.rename(columns={'sample_id': 'new_rows'})
        x5_2 = x5_1.drop(columns=['rows']).drop_duplicates()
        x5_3 = x5_1[['rows', 'new_rows']].copy()
        x5_3['w'] = x5_3.groupby('new_rows').new_rows.transform('size')
        x5_3['w'] = 1 / x5_3.w
        x5_3['rows'] = pd.Series(range(x5_1.shape[0]), index=x5_1.rows)[x5_3.rows.to_numpy()].to_numpy()
        x5_3['new_rows'] = pd.Series(range(x5_2.shape[0]), index=x5_2.new_rows)[x5_3.new_rows.to_numpy()].to_numpy()
        x5_3 = daa.from_array(
            sparse.COO(x5_3[['new_rows', 'rows']].to_numpy().T, x5_3.w.astype('float32')),
            chunks=(1000, -1)
        )
        return [x5_2, x5_3]

    @lazy_property
    def mat3(self):
        new_cols = self.mat3_cols
        new_rows = self.mat3_rows
        mat = self.mat2[['data', 'rows', 'cols']]
        mat = mat.sel(cols=new_cols.cols)
        mat = mat.merge(new_cols)
        mat = mat.sel(cols=np.isnan(mat.data).sum(axis=0)==0)
        mat = mat.swap_dims({'cols': 'new_cols'}).drop('cols').rename({'new_cols': 'cols'})
        mat = mat.merge(new_rows[0].set_index('new_rows'))
        new_rows[1] = new_rows[1].rechunk((None, mat.data.chunks[0]))
        data = new_rows[1] @ mat.data.data.astype('float32')
        data = daa.log2(data+0.1)
        mat['data'] = (('new_rows', 'cols'),  data)
        mat = mat.drop('rows').rename({'new_rows': 'rows'})
        mat = _add_locs(mat)
        return mat


cnv = CNV()

