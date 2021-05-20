import importlib
import xarray as xa
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
import plotly.express as px

import depmap_gdc_fit
importlib.reload(depmap_gdc_fit)
import depmap_gdc_fit as gdf

def plot1(m):
    plt.figure().gca().plot(sorted(m.stats.train), sorted(m.stats.rand), '.', alpha=0.1)
    plt.gcf().gca().axline(tuple([m.stats[['train', 'test']].min().min()]*2), slope=1)

def plot2(m):
    plt.figure().gca().plot(m.stats.train, m.stats.test, '.', alpha=0.1)

def plot3(d):
    px.scatter(
        d,
        x='obs', y='pred',
        hover_data=['CCLE_Name']
    ).show()

def plot4(d):
    px.scatter(
        d.reset_index(),
        x='index', y='expr',
        color='project_id', symbol='is_normal'
    ).show()

def concat(x):
    x = (y.data.copy() for y in x)
    x = (y.assign_coords({'cols': str(i) + ':' + y.cols}) for i, y in enumerate(x))
    x = xa.concat(x, 'cols')
    x = xa.Dataset().assign(data=x)
    return x

def model(x, y, z, reg):
    m1 = SimpleNamespace(
        X = x,
        Y = y,
        Z = z,
        split = m.split
    )
    m1 = gdf.model(m1, reg)
    return m1

m = gdf.merge()
m.split = m.crispr.rows
m.split['train'] = ('rows', np.random.random(m.split.rows.shape) < 0.8)
m1 = model(m.dm_expr, m.crispr, m.gdc_expr, [0.5, None])
m2 = model(concat([m.dm_expr, m.dm_cnv]), m.crispr, concat([m.gdc_expr, m.gdc_cnv]), [0.1, None])
m3 = model(m.dm_cnv, m.crispr, m.gdc_cnv, [0, np.s_[50:450]])
m4 = model(m.dm_cnv, m.dm_expr, m.dm_cnv, [0, 50])

x1 = gdf.SVD.from_data(m.dm_cnv.data.data)
x2 = x1.s.compute()
plt.plot(x2**2, '.')
(x2**2)[100]

x1 = m2.stats.set_index('cols').join(m1.stats.set_index('cols'), lsuffix='_x', rsuffix='_y')
plt.figure().gca().hist(x1.train_x-x1.train_y, 100)
plt.figure().gca().hist(x1.test_x-x1.test_y, 100)
plt.figure().gca().hist(x1.rand_x-x1.rand_y, 100)

plt.figure().gca().plot(x1.test_x, x1.test_y, '.')
plt.gcf().gca().axline(tuple([x1[['test_x', 'test_y']].min().min()] * 2), slope=1)

plt.figure().gca().plot(x1.train_x, x1.train_y, '.')
plt.gcf().gca().axline(tuple([x1[['train_x', 'train_y']].min().min()] * 2), slope=1)

plot1(m1)
plot2(m1)

plot1(m2)
plot2(m2)

plot1(m3)
plot2(m3)

plot1(m4)
plot2(m4)

plot3(m1.data(6665)[0])

plot4(m1.data(6665)[2])
