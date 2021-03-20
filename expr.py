import pandas as pd
from common.defs import pipe, lfilter, lazy_method, lazy_property, lapply
from common.dir import Dir, cached_property, cached_method
from gdc.expr import expr as _expr
from helpers import SparseMat1, SparseMat, cache
import numpy as np
import sklearn.preprocessing as sklp
import multiprocessing.pool as mp
import more_itertools as mit

def pd_split(df, sz):
    return mit.chunked(range(df.shape[0]), sz) | pipe | \
        lapply(lambda i: df.iloc[i])

def sweep(files, map, reduce):
    global _sweep1
    def _sweep1(files):
        def _sweep2(files):
            data = pd.concat((
                _expr.file_data(id) for id in files
            ))
            return map(data)
        data = pd.conca((
            _sweep2(files)
            for files in mit.chunked(files, 50)
        ))
        return reduce(data)

    with mp.Pool(20) as pool:
        data = pool.map(_sweep1, mit.chunked(files, 500))
    data = pd.concat(data)
    return reduce(data)

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

    @lazy_property
    @cached_property(type=Dir.csv)
    def genes(self):
        import multiprocessing.pool as mp
        import more_itertools as mit

        def data_summary(data):
            data = data.groupby(['Ensembl_Id', 'project_id', 'sample_type', 'is_normal']).agg({
                'n': sum,
                's': sum,
                'ss': sum
            })
            data.reset_index(inplace=True)
            return data

        global _process
        def _process(files):
            def _process1(files):
                data = (
                    _expr.file_data(file_id)
                    for file_id in files.id
                ) |pipe| pd.concat
                data = data[data.value>0]

                files = files.set_index('id')
                data.set_index('file_id', inplace=True)
                data = data.join(files)
                data['n'] = 1
                data['s'] = np.log2(data['value']+1e-3)
                data['ss'] = data['s'] ** 2
                return data_summary(data)

            files = mit.chunked(range(files.shape[0]), 50) | pipe | \
                    lapply(lambda i: files.iloc[i]) | pipe | list
            data = [_process1(files) for files in files]
            return data_summary(pd.concat(data))

        files = self.files
        files = mit.chunked(range(files.shape[0]), 500) |pipe|\
            lapply(lambda i: files.iloc[i]) |pipe| list
        with mp.Pool(20) as pool:
            data = pool.map(_process, files)

        data = data_summary(pd.concat(data))
        files = self.files
        files['m'] = 1
        files = files.groupby(['project_id', 'sample_type']).agg({'m': sum})
        data.set_index(['project_id', 'sample_type'], inplace=True)
        data = data.join(files)
        data.reset_index(inplace=True)
        return data


    def data(self, files):
        genes = self.genes
        data = (
                   _expr.file_data(file_id)
                   for file_id in files.id
               ) | pipe | pd.concat
        data = SparseMat(
            data.Ensembl_Id, data.file_id, data.value,
            genes.Ensembl_Id, files.id
        )
        data = data.mat.toarray()
        return data

    def log2_scale(self, data):
        data = np.log2(data + 1)
        fit = sklp.StandardScaler().fit(data)
        data = fit.transform(data)
        return (fit, data)

    @lazy_method(key=lambda project_id: project_id)
    @cached_method(type=Dir.pickle, key=lambda project_id: project_id)
    def normalization(self, project_id):
        files = self.files
        files = files[files.is_normal == True]
        files = files[files.project_id == project_id]
        log2_scaled = self.log2_scale(self.data(files))[1]
        normalization = sklp.StandardScaler().fit(log2_scaled.T)
        return normalization

    def normalize(self, file):
        normalization = self.normalization(file.project_id)
        log2_scaled = self.log2_scale(self.data(pd.DataFrame([file])))[1]
        normalized = normalization.transform(log2_scaled.T).T
        return normalized

expr = Expr()
expr.storage = cache.child('expr')


