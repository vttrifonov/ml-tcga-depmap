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
files = self.files
project_id = 'TCGA-COAD'




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

