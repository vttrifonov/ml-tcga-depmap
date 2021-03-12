import pandas as pd
import scipy.sparse as sparse
from common.defs import lazy_property, pipe
from common.dir import Dir, cached_property
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


class ZarrMat:
    @property
    @cached_property(type=Dir.zarr(chunks=(1, None)))
    def mat(self):
        return np.empty()

    @property
    @cached_property(type=Dir.csv)
    def rownames(self):
        return pd.DataFrame({'rownames': pd.Series(dtype='str')})

    @property
    @cached_property(type=Dir.csv)
    def colnames(self):
        return pd.DataFrame({'colnames': pd.Series(dtype='str')})

    def append(self, sparse):
        sparse_rownames = pd.DataFrame({'rownames': sparse.rownames}).astype(str)
        sparse_colnames = pd.DataFrame({'colnames': sparse.colnames}).astype(str)
        sparse_mat = sparse.mat.todense()

        rownames = self.rownames.astype(str)
        if rownames.shape[0] == 0:
            rownames = sparse_rownames
            colnames = sparse_colnames
            Dir.zarr(chunks=(1, None))(self.storage, 'mat').store(sparse_mat)
        else:
            colnames = self.colnames.astype(str)

            rownames['row1'] = range(rownames.shape[0])
            sparse_rownames['row2'] = range(sparse_rownames.shape[0])
            rownames = rownames.merge(sparse_rownames, right_on='rownames', left_on='rownames', how='outer')
            rownames = rownames.sort_values(by=['row1', 'row2'])
            rownames['row1'] = range(rownames.shape[0])
            rows = rownames[~rownames.row2.isna()].sort_values(['row2'])
            rownames = rownames[['rownames']]

            colnames['col1'] = range(colnames.shape[0])
            sparse_colnames['col2'] = range(sparse_colnames.shape[0])
            colnames = colnames.merge(sparse_colnames, right_on='colnames', left_on='colnames', how='outer')
            colnames = colnames.sort_values(by=['col1', 'col2'])
            colnames['col1'] = range(colnames.shape[0])
            cols = colnames[~colnames.col2.isna()].sort_values(['col2'])
            colnames = colnames[['colnames']]

            mat = self.mat
            mat.resize((rownames.shape[0], colnames.shape[0]))
            mat.set_orthogonal_selection(
                (rows.row1, cols.col1),
                sparse_mat
            )

        Dir.csv(self.storage, 'rownames').store(rownames)
        Dir.csv(self.storage, 'colnames').store(colnames)

class SparseMat1:
    @lazy_property
    @cached_property(type=Dir.csv)
    def row(self):
        return None

    @property
    def rownames(self):
        return self.row.name

    @property
    def namerows(self):
        return pd.Series(range(len(self.rownames)), index=self.rownames)

    @lazy_property
    @cached_property(type=Dir.csv)
    def col(self):
        return None

    @property
    def colnames(self):
        return self.col.name

    @property
    def namecols(self):
        return pd.Series(range(len(self.colnames)), index=self.colnames)

    def slice(self, i):
        return Dir.csv(self.storage.child('slice'), str(i)).restore()

    def assign(self, i, j, v):
        Dir.csv(self.storage, 'row').store(pd.DataFrame(dict(name=list(set(i)))))
        Dir.csv(self.storage, 'col').store(pd.DataFrame(dict(name=list(set(j)))))

        x = pd.DataFrame({
            'i': self.namerows[i].tolist(),
            'j': self.namecols[j].tolist(),
            'v': v
        })
        slice = self.storage.child('slice')
        slice.create = False
        if slice.exists:
            slice.remove()
        slice.create = True
        slice.exists
        for i, group in x.groupby('i'):
            Dir.csv(slice, str(i)).store(group)

class Analysis1:
    @classmethod
    def project_workflow_data(cls, project_id, workflow_type):
        print(f'{project_id} {workflow_type}')
        x = snv.project_data(project_id, workflow_type)
        return x

    def project_data(self, project_id):
        x1 = (
             self.project_workflow_data(project_id, workflow_type)
             for workflow_type in snv.project_workflows(project_id).workflow_type
         ) |pipe| pd.concat

        x2 = x1[['case_id', 'Entrez_Gene_Id', 'Variant_Classification']].drop_duplicates()
        x2 = x2.loc[~x2.Variant_Classification.isin([
            'Silent', 'Intron',
            "3'UTR", "5'UTR", "3'Flank", "5'Flank",
            'RNA', 'IGR'
        ])]
        x2 = x2[['case_id', 'Entrez_Gene_Id']].drop_duplicates()
        return SparseMat(x2.case_id, x2.Entrez_Gene_Id, [1] * x2.shape[0])

    @lazy_property
    def storage(self):
        return self.cache.child('analysis1')

    @lazy_property
    def snv(self):
        x2 = self.project_data(self.project_id)
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
            self.train = x[train,:]
            self.test = x[~train,:]

    @lazy_property
    def snv_data(self):
        return self.SNVData(self.snv)


analysis1 = Analysis1()
analysis1.cache = Dir(Path.home() / ".cache" / "ml-tcga-depmap")

