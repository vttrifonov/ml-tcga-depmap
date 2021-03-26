import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common.defs import pipe, lapply, lfilter
import seaborn as sb
import more_itertools as mit
import functools as ft
import tensorflow as tf
import ae
import analysis1
from snv import snv

importlib.reload(ae)
importlib.reload(analysis1)
import ae
import analysis1

def m_aucs(data):
    from helpers import roc
    return data |pipe|\
        lfilter(lambda x: x[0].sum()>0) |pipe|\
        lapply(lambda x: roc(x[0][0, :], x[1][0,: ])[2]) |pipe| list

a = analysis1.analysis1

cases = snv.cases
cases = cases[cases.project_id == 'TCGA-COAD'].case_id.reset_index(drop=True)
#cases = cases.case_id

d = a.snv_data(cases)
d.m = {}

d.m['pca'] = d.fit(ae.PCA(100))
d.m['pca'].ae.model.fit(mit.first(d.train.batch(sum(d.select)))[0])

d.m['ae1'] = d.fit(ae.AE(len(d.mat.colnames), 100, 'linear', 'linear', 'adam', 'mse'))
d.m["ae1"].ae.model.fit(
    d.train.batch(sum(d.select)).repeat(),
    validation_data=d.test.batch(sum(~d.select)), validation_steps=1,
    epochs=10, steps_per_epoch=1
)

d.m['ae2'] = d.fit(ae.AE(len(d.mat.colnames), 100, 'relu', 'sigmoid', 'adam', 'binary_crossentropy'))
d.m['ae2'].ae.model.fit(
    d.train.batch(sum(d.select)).repeat(),
    validation_data=d.test.batch(sum(~d.select)), validation_steps=1,
    epochs=100, steps_per_epoch=1
)

d.m['ae3'] = d.fit(ae.AE1(len(d.mat.colnames), [200, 100], [200]))
d.m['ae3'].fit(
    d.train.batch(sum(d.select)).repeat(),
    epochs=100, steps_per_epoch=1
)

d.m['ae4'] = d.fit(ae.AE3(len(d.mat.colnames), (50, 0, 0.01), 100))
d.m['ae4'].ae.model.fit(
    d.train.batch(sum(d.select)).repeat(),
    validation_data=d.test.batch(sum(~d.select)), validation_steps=1,
    epochs=100, steps_per_epoch=1
)

d.m |pipe|\
    (lambda x: ((k, v) for k, v in x.items())) |pipe|\
    lapply(lambda x: pd.DataFrame({"m": x[0], "x": m_aucs(x[1].test_decoded)})) |pipe|\
    (lambda x: (pd.concat(x),)) |pipe|\
    (lambda x: sb.kdeplot(x='x', hue='m', data=x[0]))

(
    m_aucs(d.m['pca'].test_decoded),
    m_aucs(d.m['ae2'].test_decoded)
) |pipe|\
    np.corrcoef

mit.nth(d.m['pca'].train_decoded, 0) |pipe|\
    (lambda x: (x[0][0,:], x[1][0,:])) |pipe|\
    (lambda x: plot_roc(x[0], x[1]))
    #(lambda x: sb.boxplot(x=x[0], y=x[1]))

(d.m['ae2'], d.m['ae4'], 1) |pipe|\
    (lambda x: (
        mit.nth(x[0].test_decoded, x[2])[1][0,:],
        mit.nth(x[1].test_decoded, x[2])[1][0,:]
    )) |pipe|\
    (lambda x: sb.scatterplot(x=x[0], y=x[1]))


