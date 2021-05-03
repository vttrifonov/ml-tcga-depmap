import pandas as pd
from common.defs import lazy_property, pipe, lapply
from common.dir import Dir
import numpy as np
from pathlib import Path
import more_itertools as mit
import matplotlib.pyplot as plt

cache = Dir(Path.home() / ".cache" / "ml-tcga-depmap")

def rev_index(ds):
    return pd.Series(ds.index, index=ds)

class SparseMat:
    def __init__(self, i, j, v, rownames = None, colnames = None):
        import scipy.sparse as sparse

        if rownames is None:
            rownames = pd.Series(i).drop_duplicates().reset_index(drop=True)

        if colnames is None:
            colnames = pd.Series(j).drop_duplicates().reset_index(drop=True)

        self.rownames = rownames
        self.colnames = colnames

        self.mat = sparse.coo_matrix(
            (v, (rev_index(self.rownames)[i], rev_index(self.colnames)[j])),
            shape=(len(rownames), len(colnames))
        )

class Mat1:
    @property
    def namerows(self):
        return rev_index(self.rownames)

    @property
    def namecols(self):
        return rev_index(self.colnames)

    def dense_tensor(self, names):
        import tensorflow as tf
        return tf.data.Dataset.from_generator(
            lambda: ((row, row) for row in (self.dense_row(name) for name in names)),
            output_shapes=((self.m,), (self.m,)),
            output_types=(tf.float64, tf.float64)
        )

    @lazy_property
    def dense(self):
        return np.vstack([self.dense_row(name) for name in self.rownames])

    def dense_tensor1(self, names):
        import tensorflow as tf
        return tf.data.Dataset.from_tensor_slices(self.dense[self.namerows[names],:]). \
            map(lambda row: (row, row))

    @property
    def xarray(self):
        import xarray as xa
        return xa.DataArray(
            self.dense,
            coords={'row': self.rownames, 'col': self.colnames},
            dims=['row', 'col']
        )

class Mat2(Mat1):
    default = 0

    @lazy_property
    def m(self):
        return len(self.colnames)

    def sparse_row(self, name):
        row = self.row_data(name).join(pd.DataFrame({'col': self.namecols}))
        row['row'] = name
        return row

    def dense_row(self, name):
        data = self.sparse_row(name)
        row = np.full((self.m,), self.default)
        row[data.col] = data.value
        return row

    def sparse(self, names):
        data = pd.concat((self.sparse_row(name) for name in names))
        return SparseMat(
            data.row, self.colnames[data.col], data.value,
            names, self.colnames
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


def roc(obs, pred):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(obs, pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_roc(obs, pred):
    fpr, tpr, roc_auc = roc(obs, pred)

    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2, label=f'AUC = {roc_auc:.02f}'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.show()

def slice_iter(b, e, c):
    if b % c != 0:
        b1 = c*np.ceil(b/c)
        yield slice(b, b1)
        b = b1
    for i in range(b, e, c):
        yield slice(i, min(i+c, e))