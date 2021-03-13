import pandas as pd
import scipy.sparse as sparse
from common.defs import lazy_property, pipe, lfilter, lazy_method
from common.dir import Dir, cached_property, cached_method
from gdc.snv import snv
import numpy as np
import ae
from pathlib import Path

class SparseMat:
    def __init__(self, i, j, v):
        self.rownames = list(set(i))
        self.namerows = pd.Series(range(len(self.rownames)), index=self.rownames)

        self.colnames = list(set(j))
        self.namecols = pd.Series(range(len(self.colnames)), index=self.colnames)

        self.mat = sparse.coo_matrix((v, (self.namerows[i], self.namecols[j])))

class Analysis1:
    @classmethod
    def project_workflow_data(cls, project_id, workflow_type):
        print(f'{project_id} {workflow_type}')
        x = snv.project_data(project_id, workflow_type)
        return x

    def project_data(self, project_id):
        x1 = (
             self.project_workflow_data(project_id, workflow_type)
             for workflow_type in snv.project_workflows(project_id)
         ) |pipe| pd.concat

        x2 = x1[['case_id', 'Entrez_Gene_Id', 'Variant_Classification']].drop_duplicates()
        x2 = x2.loc[~x2.Variant_Classification.isin([
            'Silent', 'Intron',
            "3'UTR", "5'UTR", "3'Flank", "5'Flank",
            'RNA', 'IGR'
        ])]
        x2 = x2[['case_id', 'Entrez_Gene_Id']].drop_duplicates()
        x2 = SparseMat(x2.case_id, x2.Entrez_Gene_Id, [1] * x2.shape[0])

        return x2

    @lazy_property
    def storage(self):
        return self.cache.child('analysis1')

    def snv(self, project_id):
        x2 = self.project_data(project_id)
        x2.row_stats = pd.DataFrame({
            "case_id": x2.rownames,
            "mean": np.asarray(x2.mat.mean(axis=1)).flatten(),
        })
        x2.col_stats = pd.DataFrame({
            'Entrez_Gene_Id': x2.colnames,
            "mean": np.asarray(x2.mat.mean(axis=0)).flatten(),
        })
        x2.analysis = self
        return x2

    class SNVData(ae.Data):
        def __init__(self, snv):
            self.snv = snv
            train = np.asarray(snv.mat.todense())
            x = train[:, train.sum(axis=0)>2]
            train = np.random.random(x.shape[0])<0.7
            self.train1 = x[train,:]
            self.train = x[train,:]
            self.test = x[~train,:]

    def snv_data(self, project_id):
        return self.SNVData(self.snv(project_id))


    class SNV1:
        @property
        def cases(self):
            m = snv.manifest
            return m[m.project_id.str.match('TCGA')][['project_id', 'case_id']].\
                drop_duplicates().\
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
            m = snv.manifest
            m = m[m.case_id == case_id]
            x1 = (
                snv.case_data(workflow_type, case_id)
                for workflow_type in m.workflow_type
            ) |pipe|\
                lfilter(lambda x: x is not None) |pipe| list

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

        class IterCases:
            @lazy_property
            def m(self):
                return len(self.genes)

            @lazy_property
            def idx(self):
                return pd.Series(range(self.m), index=self.genes)

            @lazy_method(key=lambda case_id: case_id)
            def case_data(self, case_id):
                g = self.snv.case_data(case_id).Entrez_Gene_Id
                g = g[g.isin(self.genes)]
                return self.idx[g]

            def iter(self, cases):
                for case_id in cases:
                    t = np.full((self.m,), 0)
                    t[self.case_data(case_id)] = 1
                    yield t, t

            def tensor(self, cases):
                import tensorflow as tf
                return tf.data.Dataset.from_generator(
                    lambda: self.iter(cases),
                    output_shapes=((self.m,), (self.m,)),
                    output_types=(tf.float64, tf.float64)
                )

    @lazy_property
    def snv1(self):
        snv1 = self.SNV1()
        snv1.storage = self.storage.child('snv1')
        snv1.analysis = self
        return snv1

    class SNV1Data(ae.Data):
        def __init__(self, snv, cases, genes):
            if cases is None:
                cases = snv.cases.case_id
            if genes is None:
                genes = snv.genes.Entrez_Gene_Id

            iter = snv.IterCases()
            iter.genes = genes
            iter.snv = snv

            self.iter = iter

            train = np.random.random(len(cases))<0.7
            self.n_train = sum(train)
            self.cases = cases
            self.genes = genes
            self.train1 = iter.tensor(cases[train]).batch(sum(train))
            self.train = iter.tensor(cases[train]).repeat().batch(sum(train))
            self.test = iter.tensor(cases[~train]).batch(sum(~train))


    def snv1_data(self, cases = None, genes = None):
        return self.SNV1Data(self.snv1, cases, genes)

analysis1 = Analysis1()
analysis1.cache = Dir(Path.home() / ".cache" / "ml-tcga-depmap")

