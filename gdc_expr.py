import pandas as pd
from common.defs import pipe, lazy_property
from common.dir import Dir, cached_property, cached_method
from gdc.expr import expr as _expr
from helpers import Mat1, map_reduce, slice_iter, config
import numpy as np
import zarr
from joblib import Parallel, delayed
from pathlib import Path
import itertools as it
import numcodecs as nc
from pathlib import Path
import dask.array as daa
import dask_ml.preprocessing as dmlp
import dask_ml.decomposition as dmld
import xarray as xa
import ensembl.sql.ensembl as ensembl

class Expr:
    @lazy_property
    def storage(self):
        return Dir(config.cache/'gdc/expr')

    @lazy_property
    @cached_property(type=Dir.pickle)
    def files(self):
        import json
        m = _expr.manifest
        m = m[m.project_id.str.match('TCGA')]
        m = m[m.workflow_type.isin(['HTSeq - FPKM'])]
        m['sample_id'] = [json.loads(sample)[0]['sample_type'] for sample in m.samples]
        m['sample_type'] = [json.loads(sample)[0]['sample_type'] for sample in m.samples]
        m['is_normal'] = m.sample_type.str.contains('Normal')
        return m[[
            'project_id', 'id',
            'case_id', 'sample_id', 'sample_type',
            'is_normal']]. \
            reset_index(drop=True)

    @property
    def cases(self):
        return self.files[['project_id', 'case_id']]. \
            drop_duplicates(). \
            reset_index(drop=True)

    def data(self, file):
        return _expr.file_data(file)

    @lazy_property
    @cached_property(type=Dir.csv)
    def genes(self):
        files = self.files.set_index('id')

        def reduce(data):
            return pd.concat(data).\
                groupby(['Ensembl_Id', 'project_id', 'sample_type', 'is_normal']).\
                agg({'n': sum, 's': sum, 'ss': sum}).\
                reset_index()

        def map(file):
            data = self.data(file).\
                set_index('file_id').\
                join(files)
            data['n'] = 1
            data['s'] = np.log2(data.value + 1e-3)
            data['ss'] = data.s ** 2
            return data

        return list(files.index) |pipe| map_reduce(map, reduce)

    @lazy_property
    def normalization(self):
        genes = self.genes.copy()
        genes['s'] /= genes.n  # mean
        genes['ss'] -= genes.n * genes.s ** 2  # (n-1)*var
        genes.set_index(['project_id', 'Ensembl_Id'], inplace=True)
        genes = genes.add_prefix('t_').join(genes.query('is_normal').add_prefix('n_'))
        genes.reset_index(inplace=True)
        del genes['n_is_normal']
        del genes['n_sample_type']
        genes.rename(columns={'t_sample_type': 'sample_type'}, inplace=True)
        genes['se'] = genes.n_ss + genes.t_ss
        genes['se'] /= genes.n_n + genes.t_n - 2
        genes['se'] *= 1 / genes.n_n + 1 / genes.t_n
        genes['se'] **= 0.5
        genes.set_index(['project_id', 'sample_type', 'Ensembl_Id'], inplace=True)
        return genes

    def normalized_data1(self, file):
        data = self.data(file).\
            set_index('file_id').\
            join(
                self.files.\
                    rename(columns={'id': 'file_id'}).\
                    set_index('file_id')[['project_id', 'sample_type']]
            ).\
            reset_index().\
            set_index(['project_id', 'sample_type', 'Ensembl_Id']).\
            join(self.normalization[['t_s', 'se']]).\
            reset_index()
        data['value'] = np.log2(data.value + 1e-3)
        data['value'] -= data.t_s
        data['value'] /= data.se
        return data[['file_id', 'Ensembl_Id', 'value']]

    @cached_method(type=Dir.pickle, key=lambda file: file)
    def normalized_data(self, file):
        data = self.normalized_data1(file).sort_values('Ensembl_Id').value
        return np.float16(data)

    def _normalize_all(self):
        self.normalization
        self.storage.child('normalized_data').exists
        list(self.files.id) | pipe | \
            map_reduce(
                map=lambda file: (self.normalized_data(file), None)[1],
                reduce=lambda _: (list(_), None)[1]
            )

    class FullMat(Mat1):
        def __init__(self, expr):
            self.expr = expr

        @lazy_property
        def storage(self):
            return self.expr.storage.child('full_mat')

        @lazy_property
        def rownames(self):
            return self.expr.files.id.reset_index(drop=True)

        @lazy_property
        def colnames(self):
            return  self.expr.genes.Ensembl_Id.drop_duplicates().sort_values().reset_index(drop=True)

        def dense_row(self, name):
            return self.expr.normalized_data(name)

        @lazy_property
        @cached_property(type=Dir.pickle)
        def dense(self):
            return super().dense

    @lazy_property
    def full_mat(self):
        return self.FullMat(self)

    class Mat(Mat1):
        def __init__(self, expr, genes):
            self.colnames = genes
            self.expr = expr

        @property
        def rownames(self):
            return self.expr.full_mat.rownames

        @lazy_property
        def dense(self):
            full = self.expr.full_mat
            return np.nan_to_num(full.dense)[:, full.namecols[self.colnames]]

        def dense_row(self, name):
            return self.dense[self.namerows[name],:]

        def tensor(self, cases):
            return self.dense_tensor1(cases)

    def mat(self, genes):
        return self.Mat(self, genes)

    class Mat2:
        def __init__(self, expr):
            self.expr = expr

        @lazy_property
        def rows(self):
            return self.expr.files.\
                set_index('id').\
                sort_values(['project_id', 'sample_type', 'is_normal'])

        @lazy_property
        @cached_property(type=Dir.pickle)
        def cols(self):
            rows = self.rows.index

            def _colnames(_range):
                colnames = set()
                slices = slice_iter(_range.start, _range.stop, 100)
                for _slice in slices:
                    data = (self.expr.data(file).Ensembl_Id for file in rows[_slice])
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
                        data = (np.log2(self.expr.data(file).set_index('Ensembl_Id').value+1e-3) for file in rows[_slice])
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

    @lazy_property
    def mat2(self):
        mat = self.Mat2(self)
        mat.storage = self.storage.child('mat2')
        return mat

expr = Expr()

