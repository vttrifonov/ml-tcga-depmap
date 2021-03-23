import pandas as pd
from common.defs import lazy_property, pipe, lapply
from common.dir import Dir
import numpy as np
from pathlib import Path
import more_itertools as mit

cache = Dir(Path.home() / ".cache" / "ml-tcga-depmap")

class SparseMat:
    def __init__(self, i, j, v, rownames = None, colnames = None):
        import scipy.sparse as sparse

        if rownames is None:
            rownames = pd.Series(i).drop_duplicates()

        if colnames is None:
            colnames = pd.Series(j).drop_duplicates()

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

    def sparse_tensor(self, names):
        import tensorflow as tf
        def map(i, v, s):
            sparse = tf.SparseTensor(i, v, dense_shape=s)
            sparse = tf.sparse.reorder(sparse)
            return (sparse, tf.sparse.to_dense(sparse))
        return tf.data.Dataset.from_generator(
            lambda: ((
                np.array([row.col]).T,
                row.value,
                [self.m]
            ) for row in (self.sparse_row(name) for name in names)),
            output_types=(tf.int64, tf.float64, tf.int64)
        ).map(map)

processes = 10
def par_lapply(f):
    import multiprocessing.pool as mp

    global __f
    def __f(x):
        return f(x)

    def lapply(l):
        with mp.Pool(processes) as pool:
            return pool.map(__f, l)
    return lapply

def conc_lapply(threads, f):
    import multiprocessing.pool as mp

    global __f
    def __f(x):
        return f(x)

    def lapply(l):
        with mp.ThreadPool(threads) as pool:
            return pool.map(__f, l)
    return lapply


def map_reduce(map, reduce, par_batch = None, conc_batch = None, conc_threads = 1):
    def map_reduce(input):
        nonlocal par_batch, conc_batch, conc_threads

        if par_batch is None:
            input = list(input)
            par_batch = np.ceil(len(input) / processes)
        if conc_batch is None:
            conc_batch = par_batch
        par_batch = int(par_batch)
        conc_batch = int(conc_batch)

        return mit.chunked(input, par_batch) |pipe|\
            par_lapply(
                lambda input:
                    mit.chunked(input, conc_batch) |pipe|\
                        conc_lapply(
                            conc_threads,
                            lambda input:
                                input |pipe|\
                                    lapply(map) |pipe|\
                                    reduce
                        ) |pipe|\
                        reduce
            ) |pipe|\
            reduce

    return map_reduce