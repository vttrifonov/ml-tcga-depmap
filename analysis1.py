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

class SparseMat1:
    default = 0

    def __init__(self, colnames):
        self.colnames = colnames

    @lazy_property
    def m(self):
        return len(self.colnames)

    @lazy_property
    def namecols(self):
        return pd.DataFrame({
            'col': range(len(self.colnames))
        }, index=self.colnames)

    def sparse_row(self, name):
        row = self.row_data(name).join(self.namecols)
        row['row'] = name
        return  row

    def dense_row(self, name):
        row = np.full((self.m,), self.default)
        data = self.sparse_row(name)
        row[data.col] = data.value
        return row

    def sparse(self, names):
        data = pd.concat((self.sparse_row(name) for name in names))
        return SparseMat(data.row, data.col, data.value)

    def tensor(self, names):
        import tensorflow as tf
        return tf.data.Dataset.from_generator(
            lambda: ((row, row) for row in (self.dense_row(name) for name in names)),
            output_shapes=((self.m,), (self.m,)),
            output_types=(tf.float64, tf.float64)
        )

class Analysis1:
    @lazy_property
    def storage(self):
        return self.cache.child('analysis1')

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

        class Mat(SparseMat1):
            def __init__(self, snv, genes):
                super().__init__(genes)
                self.snv = snv

            @lazy_method(key=lambda name: name)
            def sparse_row(self, name):
                return super().sparse_row(name)

            def row_data(self, name):
                g = self.snv.case_data(name).Entrez_Gene_Id
                return pd.DataFrame({'value': [1]*len(g)}, index=g)

        def mat(self, genes):
            return self.Mat(self, genes)

    @lazy_property
    def snv1(self):
        snv1 = self.SNV1()
        snv1.storage = self.storage.child('snv1')
        snv1.analysis = self
        return snv1

    class SNVData(ae.Data):
        def __init__(self, snv, cases, genes):
            if cases is None:
                cases = snv.cases.case_id
            if genes is None:
                genes = snv.genes.Entrez_Gene_Id

            self.mat = snv.mat(genes)

            train = self.mat.sparse(cases).mat.toarray()

            x = train[:, train.sum(axis=0)>2]
            train = np.random.random(x.shape[0])<0.7
            self.train1 = x[train,:]
            self.train = x[train,:]
            self.test = x[~train,:]

    def snv_data(self, cases = None, genes = None):
        return self.SNVData(self.snv1, cases, genes)

    class SNV1Data(ae.Data):
        def __init__(self, snv, cases, genes):
            if cases is None:
                cases = snv.cases.case_id
            if genes is None:
                genes = snv.genes.Entrez_Gene_Id

            self.mat = snv.mat(genes)

            train = np.random.random(len(cases))<0.7
            self.n_train = sum(train)
            self.cases = cases
            self.train1 = self.mat.tensor(cases[train]).batch(sum(train))
            self.train = self.mat.tensor(cases[train]).repeat().batch(sum(train))
            self.test = self.mat.tensor(cases[~train]).batch(sum(~train))


    def snv1_data(self, cases = None, genes = None):
        return self.SNV1Data(self.snv1, cases, genes)

analysis1 = Analysis1()
analysis1.cache = Dir(Path.home() / ".cache" / "ml-tcga-depmap")

