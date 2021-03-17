import pandas as pd
from common.defs import pipe, lfilter, lazy_method
from common.dir import Dir, cached_property, cached_method
from gdc.snv import snv as _snv
from helpers import SparseMat1, cache

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

    class Mat(SparseMat1):
        def __init__(self, snv, genes):
            super().__init__(genes)
            self.snv = snv

        @lazy_method(key=lambda name: name)
        def sparse_row(self, name):
            return super().sparse_row(name)

        def row_data(self, name):
            g = self.snv.case_data(name).Entrez_Gene_Id
            return pd.DataFrame({'value': [1] * len(g)}, index=g)

    def mat(self, genes):
        return self.Mat(self, genes)

snv = SNV()
snv.storage = cache.child('snv')