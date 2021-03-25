import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common.defs import pipe, lapply, lfilter
import seaborn as sb
import more_itertools as mit
import functools as ft
import expr

importlib.reload(expr)
import expr

self = expr.expr


def _():
    import multiprocessing as mp
    import ctypes as ct
    import numpy as np

    shared_array = None

    def init(shared_array_base):
        global shared_array
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(10, 10)

    # Parallel processing
    def my_func(i):
        shared_array[i, :] = i

    shared_array_base = mp.Array(ct.c_double, 10 * 10)

    pool = mp.Pool(processes=4, initializer=init, initargs=(shared_array_base,))
    pool.map(my_func, range(10))

    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(10, 10)
    print(shared_array)


def _():
    from helpers import map_reduce
    from time import time

    self._normalize_all()
    file = self.files.sample(1).id.iloc[0]
    self.normalized_data(file)

    mat = self.mat(self.gene_index.index)

    t0 = time()
    m = list(self.files.id) |pipe|\
        map_reduce(
            map = mat.dense_row,
            reduce = list
        )
    print(time() - t0)

    m.squeeze().reshape(100, 60483).shape

    from time import time
    t0 = time()
    x = [mat.dense_row(name) for name in self.files.id]
    print(time() - t0)

    x = np.vstack(x)
    import pickle
    with open('tmp.1.pickle', "wb") as file:
        pickle.dump(x, file)

    t0 = time()
    with open('tmp.1.pickle', "rb") as file:
        x = pickle.load(file)
    print(time() - t0)


def _():
    import tensorflow as tf
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    import scipy.sparse as sparse


    x1 = np.random.randn(15).reshape(3, 5)
    x2 = layers.Dense(5, kernel_initializer=keras.initializers.Constant(x1))
    x6_1 = sparse.csr_matrix(
        ([1, 2, 3], ([0, 0, 1], [0, 2, 1])),
        shape=(2, 3)
    )
    def x6_2():
        for i in range(x6_1.shape[0]):
            r = x6_1[i,:]
            yield (
                np.array([r.indices]).T,
                r.data,
                [x6_1.shape[1]]
            )
    x6_3 = tf.data.Dataset.from_generator(x6_2, output_types=(tf.int64, tf.float64, tf.int64)).\
        map(tf.SparseTensor).batch(2)

    mit.first(x6_3).shape

    x2(mit.first(x6_3)) - x6_1 @ x1

def _():
    def f1(data):
        return pd.concat(data).groupby('Ensembl_Id').sum()

    def f2(data):
        return f1([data.set_index('Ensembl_Id').values])

    files  = self.files

    import time
    t0 = time.time()
    sweep(files[0:5000].id, f2, f1)
    print(time.time()-t0)

    d = pd.DataFrame({
        'n': [1000, 1500, 2000, 3000, 3500, 4000, 5000],
        't': [171,  187,  214,  250,  279, 313, 442]
    })

    import sklearn.linear_model as skllm

    x1 = skllm.LinearRegression()
    x2 = x1.fit(
        np.array(d.n).reshape(-1, 1),
        np.array(d.t).reshape(-1, 1)
    )

    x2.predict(np.array([[10000]]))

    plt.scatter(d.n, d.t)
    plt.scatter(d.n, x2.predict(np.array(d.n).reshape(-1, 1)))

def _():
    x2 = self.genes.copy()
    x2[['project_id', 'sample_type', 'm']].drop_duplicates().query('m>=10').sort_values('m')
    x2 = x2[x2.m>1]

    eps = np.log2(1e-3)
    x2['s'] = x2.s + (x2.m - x2.n) * eps
    x2['ss'] = x2.ss + (x2.m - x2.n) * eps**2
    x2['mean'] = x2.s / x2.m
    x2['mean^2'] = x2['mean'] ** 2
    x2['var'] = (x2.ss - x2.m * x2['mean^2']) / (x2.m - 1)
    x2['sd'] = np.sqrt(x2['var'])
    x2['cv2'] = x2['var'] / x2['mean^2']
    x2['disp'] = x2['cv2'] - 1 / x2['mean']
    x2['filter'] = x2['mean'] + x2['sd'] > 1

    x2['filter'].value_counts()

    x2.groupby('Ensembl_Id').agg({'filter': sum}).query('filter > 0')

    sb.scatterplot(
        x='mean', y='sd', data=x2.sample(100000),
        alpha=0.1, hue='sample_type'
    )

def _():
    x1 = self.genes.copy()

    files = self.files.copy()
    x2 = x1.groupby('Ensembl_Id').agg({
        's': sum, 'ss': sum, 'n': sum
    })
    eps = np.log2(1e-3)
    x2['m'] = files.shape[0]
    x2['s'] = x2.s + (x2.m - x2.n)*eps
    x2['ss'] = x2.ss + (x2.m - x2.n)*eps**2
    x2['mean'] = x2.s/x2.m
    x2['mean^2'] = x2['mean']**2
    x2['var'] = (x2.ss - x2.m*x2['mean^2'])/(x2.m-1)
    x2['sd'] = np.sqrt(x2['var'])
    x2['cv2'] = x2['var'] / x2['mean^2']
    x2['disp'] = x2['cv2'] - 1/x2['mean']
    x2['filter'] = x2['mean'] + x2['sd'] < 0

    x2['filter'].value_counts()

    sb.scatterplot(
        x='mean', y='sd', data=x2,
        alpha=0.1, hue='filter'
    )

def _():
    import sklearn.preprocessing as sklp
    import sklearn.decomposition as skld

    files = self.files
    project_id = 'TCGA-COAD'

    f1 = files[files.project_id == project_id]
    f1 = f1[f1.workflow_type == 'HTSeq - FPKM']
    f1['col'] = range(f1.shape[0])

    data = self.data(f1)
    z = data.sum(axis=1)==0

    x1 = np.log10(data+1e-3).T

    x2 = x1[f1[f1.is_normal].col,:]
    x3 = x1[f1[~f1.is_normal].col,:]

    z2 = (x2 > x2.min()).sum(axis=0) == 0
    z3 = (x3 > x3.min()).sum(axis=0) == 0

    x2_1 = skld.PCA(5).fit(x2)
    x3_1 = skld.PCA(5).fit(x3)

    x2_2 = sklp.StandardScaler().fit(x2)

    x2_3 = x2_2.transform(x2)
    x3_3 = x2_2.transform(x3)

    plt.hist(x2_3[:, ~z2].flatten(), bins=100)

def _():
    files = self.files
    project_id = 'TCGA-COAD'

    file = files[files.project_id == project_id].iloc[1]

    data = self.data(pd.DataFrame([file]))
    log2_scaled = self.log2_scale(data)
    normalized = self.normalize(file)

    df = pd.DataFrame({
        'data': np.log2(data.flatten()+1),
        'log2_scaled': log2_scaled[1].flatten(),
        'normalized': normalized.flatten()
    })
    df['z'] = df.data==0

    g = sb.FacetGrid(df, hue='z')
    g.map(sb.histplot, 'log2_scaled', bins=100)

