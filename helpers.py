import pandas as pd
from common.defs import lazy_property
from common.dir import Dir
import numpy as np
from pathlib import Path

cache = Dir(Path.home() / ".cache" / "ml-tcga-depmap")

class SparseMat:
    def __init__(self, i, j, v, rownames, colnames):
        import scipy.sparse as sparse

        self.rownames = rownames
        self.namerows = pd.Series(range(len(rownames)), index=rownames)

        self.colnames = colnames
        self.namecols = pd.Series(range(len(colnames)), index=colnames)

        self.mat = sparse.coo_matrix(
            (v, (self.namerows[i], self.namecols[j])),
            shape=(len(rownames), len(colnames))
        )

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
        return row

    def dense_row(self, name):
        row = np.full((self.m,), self.default)
        data = self.sparse_row(name)
        row[data.col] = data.value
        return row

    def sparse(self, names):
        data = pd.concat((self.sparse_row(name) for name in names))
        return SparseMat(
            data.row, self.colnames[data.col], data.value,
            names, self.colnames
        )

    def tensor(self, names):
        import tensorflow as tf
        return tf.data.Dataset.from_generator(
            lambda: ((row, row) for row in (self.dense_row(name) for name in names)),
            output_shapes=((self.m,), (self.m,)),
            output_types=(tf.float64, tf.float64)
        )