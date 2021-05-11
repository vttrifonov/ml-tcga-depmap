import pandas as pd
from common.defs import pipe, lfilter, lazy_method, lazy_property
from common.dir import Dir, cached_property, cached_method
from gdc.snv import snv as _snv
from helpers import Mat1, Mat2, cache
import numpy as np

class SNV:
    @property
    def cases(self):
        m = _snv.manifest
        return m[m.project_id.str.match('TCGA')][['project_id', 'case_id']]. \
            drop_duplicates(). \
            reset_index(drop=True)

    @property
    @cached_property(type=Dir.csv)
    def genes(self):
        global case_genes

        import multiprocessing as mp

        def case_genes(case_id):
            print(f'{case_id}')
            return set(self.case_data(case_id).Entrez_Gene_Id)

        with mp.Pool(10) as pool:
            g = pool.map(case_genes, self.cases.case_id)

        g = list(set([]).union(*g))
        g.sort()

        return pd.DataFrame({'Entrez_Gene_Id': g})

    @cached_method(type=Dir.csv, key=lambda case_id: f'{case_id}')
    def case_data(self, case_id):
        m = _snv.manifest
        m = m[m.case_id == case_id]
        x1 = (
                 _snv.case_data(workflow_type, case_id)
                 for workflow_type in m.workflow_type
             ) | pipe | \
             lfilter(lambda x: x is not None) | pipe | list

        if not x1:
            return pd.DataFrame({'Entrez_Gene_Id': []})

        x1 = pd.concat(x1)
        x2 = x1[['case_id', 'Entrez_Gene_Id', 'Variant_Classification']].drop_duplicates()
        x2 = x2.loc[~x2.Variant_Classification.isin([
            'Silent', 'Intron',
            "3'UTR", "5'UTR", "3'Flank", "5'Flank",
            'RNA', 'IGR'
        ])]
        x2 = x2[['Entrez_Gene_Id']].drop_duplicates()
        return x2

    class FullMat(Mat2):
        default = np.int8(0)

        def __init__(self, snv):
            self.snv = snv

        @lazy_property
        def storage(self):
            return self.snv.storage.child('full_mat')

        @lazy_property
        def rownames(self):
            return self.snv.cases.case_id.reset_index(drop=True)

        @lazy_property
        def colnames(self):
            return self.snv.genes.Entrez_Gene_Id.reset_index(drop=True)

        @lazy_method(key=lambda name: name)
        def sparse_row(self, name):
            return super().sparse_row(name)

        def row_data(self, name):
            g = self.snv.case_data(name).Entrez_Gene_Id
            return pd.DataFrame({'value': [1] * len(g)}, index=g).astype('int8')

        @lazy_property
        @cached_property(type=Dir.pickle)
        def dense(self):
            return super().dense

    @lazy_property
    def full_mat(self):
        return self.FullMat(self)

    class Mat(Mat1):
        def __init__(self, snv, genes):
            self.colnames = genes
            self.snv = snv

        @property
        def rownames(self):
            return self.snv.full_mat.rownames

        @property
        def dense(self):
            full = self.snv.full_mat
            return full.dense[:, full.namecols[self.colnames]]

        @lazy_property
        def dense_row(self, name):
            return self.dense[self.namerows[name],:]

        def tensor(self, cases):
            return self.dense_tensor1(cases)

    def mat(self, genes):
        return self.Mat(self, genes)

snv = SNV()
snv.storage = cache.child('gdc/snv')