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

m = gdf.merge()
m.split = m.crispr.mat.rows
m.split['train'] = ('rows', np.random.random(m.split.rows.shape) < 0.8)
ms = SimpleNamespace(
    x = SimpleNamespace(
        dm_expr = gdf.model.splitx(m.dm_expr, m.split),
        dm_cnv = gdf.model.splitx(m.dm_cnv, m.split),
        dm_expr_cnv = gdf.model.splitx(gdf.Mat(lambda: gdf.concat([m.dm_expr.mat, m.dm_cnv.mat])), m.split)
    ),
    y = SimpleNamespace(
        crispr = gdf.model.splity(m.crispr, m.split),
        dm_expr = gdf.model.splity(m.dm_expr, m.split)
    )
)

m1 = gdf.model(ms.x.dm_expr, ms.y.crispr, m.gdc_expr.mat, [0, np.s_[:400]])
m2 = gdf.model(ms.x.dm_expr_cnv, ms.y.crispr, gdf.concat([m.gdc_expr.mat, m.gdc_cnv.mat]), [0, np.s_[:400]])
m3 = gdf.model(ms.x.dm_cnv, ms.y.crispr, m.gdc_cnv.mat, [0, np.s_[:400]])
m4 = gdf.model(ms.x.dm_cnv, ms.y.dm_expr, m.dm_cnv.mat, [0, np.s_[:400]])

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

