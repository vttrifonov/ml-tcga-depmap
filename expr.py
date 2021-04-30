import pandas as pd
from common.defs import pipe, lazy_property
from common.dir import Dir, cached_property, cached_method
from gdc.expr import expr as _expr
from helpers import Mat1, cache, map_reduce
import numpy as np

class Expr:
    @lazy_property
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


expr = Expr()
expr.storage = cache.child('expr')


