import matplotlib.pyplot as plt
import seaborn as sns
from common.dir import cached_property, Dir
from common.defs import lazy_property
from types import SimpleNamespace
import types
import dask.array as daa
import zarr
from pathlib import Path
import xarray as xa
import pickle
import numpy as np
import pandas as pd
import importlib
import dask_ml.preprocessing as dmlp
import plotly.express as px
import dask
from helpers import config
from svd import SVD
from merge import merge

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

config.exec()

def _smooth(x, window_len=11, window='hanning'):
    if len(x) < window_len:
        return x

    if window_len<3:
        return x

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    y = y[window_len-1:-(window_len-1)]
    return y

def _fft(x):
    fft = daa.hstack([x, daa.fliplr(x[:, 1:])]).rechunk((None, -1))
    fft = daa.fft.rfft(fft)
    fft = daa.real(fft)
    return fft

def _rfft(x, n):
    rfft = x[:, :n]
    if x.shape[1] > n:
        rfft = daa.hstack([rfft, daa.zeros((x.shape[0], x.shape[1] - n))])
    rfft = rfft.rechunk((None, -1))
    rfft = daa.fft.irfft(rfft)
    rfft = rfft[:, :x.shape[1]]
    return rfft

def _fft1(d1, cutoff, g, cols):
    f = xa.Dataset()

    d = d1.groupby(g)
    d = [
        xa.apply_ufunc(
            _fft, x,
            input_core_dims = [[cols]], output_core_dims=[['freq']],
            dask='allowed'
        ).assign_coords({
            'fft_'+g: ('freq', x[g]),
            'fft_freq': ('freq', np.arange(x.shape[1]))
        }).assign_coords({
            'freq': lambda x: ('freq', x['fft_'+g] + ':' + x.fft_freq.astype(str))
        })
        for _, x in d
    ]
    d = xa.concat(d, 'freq')
    d.data = d.data.rechunk(d1.data.chunks)
    f['fft'] = d
    print('hi1')

    d = f.fft.groupby('fft_' + g)
    d = [
        xa.apply_ufunc(
            _rfft, x, min(cutoff, x.shape[1]),
            input_core_dims = [['freq'], []], output_core_dims=[[cols]],
            dask='allowed'
        ).assign_coords({
            cols: d1[cols].sel({cols: d1[g]==l})
        })
        for l, x in d
    ]
    d = xa.concat(d, cols)
    d = d.sel({cols: d1.cols})
    d.data = d.data.rechunk(d1.data.chunks)
    f['rfft'] = d
    f['fft_resid'] = d1 - f.rfft
    print('hi2')

    return f

def _fft2(x1, g):
    x2 = x1[g].to_numpy()
    x2 = np.hstack([True, x2[1:]!=x2[:-1]])
    x3 = x1.index.to_numpy()
    x3 = np.hstack([True, x3[1:]!=(x3[:-1]+1)])
    return  np.cumsum(x2 | x3)
    #x1['f'] = np.cumsum(x2 | x3)
    #x4 = np.zeros(d1[cols].shape[0])
    #x4[x1.index] = x1.f
    #f['fft_group'] = (cols, x4.astype(int))
    #print('hi3')

def _svd1(d, cutoff, cols, rand = False):
    x3 = d.fft_resid
    x3 = x3.assign_coords(fft_group=d.fft_group)
    x3 = x3.sel({cols: ~x3.fft_group.isnull()})
    x4 = d.fft.sel(freq=d.fft.fft_freq<cutoff)
    x4 = x4.rename({'freq': 'cols'}).drop(['fft_arm', 'fft_freq'])
    x4['fft_group'] = ('cols', np.zeros(x4.shape[1]))
    x3 = xa.concat([x3, x4], 'cols')
    #x3 = dmlp.StandardScaler().fit_transform(x3)
    x3['fft_group'] = x3.fft_group.astype('int')
    if rand:
        x3.data = daa.apply_along_axis(np.random.permutation, 0, x3.data, shape=(x3.shape[0],), dtype=x3.dtype)
    x3 = x3.groupby('fft_group')
    x3 = {k: SVD.from_mat(x) for k, x in x3}
    return x3

def _fit(x1, x3, cutoff):
    x3 = SVD.from_mat(x3).persist()
    x2 = x3.cut(np.s_[:cutoff]).u
    x2 = ((x1 - x2 @ (x2.T @ x1)) ** 2).mean(axis=0, keepdims=True)
    x2 = daa.log10(1 - x2 + 1e-7)
    x4 = x3.perm.cut(np.s_[:cutoff]).u
    x4 = ((x1 - x4 @ (x4.T @ x1)) ** 2).mean(axis=0, keepdims=True)
    x4 = daa.log10(1 - x4 + 1e-7)
    x4 = dmlp.StandardScaler().fit(x4.T)
    x2 = x4.transform(x2.T).T.squeeze().persist()
    f = xa.Dataset()
    f['rand_mean'] = x4.mean_[0]
    f['rand_var'] = x4.var_[0]
    f['r2'] = x2
    return f

def _smooth1(d1, frac, g, cols):
    #d = m.dm_cnv.data.assign_coords(arm=m.dm_cnv.arm)
    d = d1.groupby(g)
    d = d.map(lambda x: (
        xa.apply_ufunc(
            _smooth, x, int(frac*x.shape[1]),
            input_core_dims = [[cols], []], output_core_dims=[[cols]],
            dask='parallelized', vectorize=True
        )
    ))
    d = d.drop(g)
    d.data = d.data.rechunk((None, -1))

    f = xa.Dataset()
    f['smooth'] = d
    f['smooth_resid'] = d1 - d
    return f

def _scale(d):
    fit = dmlp.StandardScaler().fit(d)
    d['mean'] = (d.dims[1], fit.mean_)
    d['var'] = (d.dims[1], fit.var_)
    d = fit.transform(d)
    return d

def plot_fft_resid(x1):
    px.scatter(
        (x1.fft_resid ** 2).mean(axis=0).rename('mean'). \
            assign_coords(arm=x1.arm, cyto=x1.cyto). \
            to_dataframe().reset_index().reset_index(). \
            query('mean>=0')
        ,
        x='index', y='mean', color='arm',
        hover_data=['cyto', 'cols']
    ).show()

def plot_fft(x1):
    px.scatter(
        x1.rfft[0, :]. \
            assign_coords(
            data=x1.data[0, :],
            arm=x1.arm, cyto=x1.cyto
        ). \
            to_dataframe().reset_index().reset_index(),
        x='index', y=['rfft', 'data'], color='arm',
        hover_data=['cyto', 'cols']
    ).show()

class _playground11:
    pass
playground11 = _playground11()

def _():
    _playground11.storage = Dir(config.cache/'playground11'/'full')

    @lazy_property
    @cached_property(type=Dir.pickle)
    def train_split_ratio(self):
        return 1
    _playground11.train_split_ratio = train_split_ratio
    #del self.__lazy__train_split_ratio

    @lazy_property
    @cached_property(type=Dir.pickle)
    def train_split(self):
        rows = merge.crispr.rows
        rows['train'] = ('rows', np.random.random(rows.shape[0])<=self.train_split_ratio)
        return rows
    _playground11.train_split = train_split
    #del self.__lazy__train_split

    @lazy_property
    def crispr(self):
        d = merge.crispr.copy()
        d = d.sel(rows=self.train_split.train)
        d['data'] = _scale(d.data.astype('float32'))
        return d
    _playground11.crispr = crispr
    #del self.__lazy__crispr

    @lazy_property
    def dm_expr(self):
        d = merge.dm_expr.copy()
        d = d.sel(rows=self.train_split.train)
        d['data'] = _scale(d.data.astype('float32'))
        return d
    _playground11.dm_expr = dm_expr
    #del self.__lazy__dm_expr

    @lazy_property
    def dm_cnv(self):
        d = merge.dm_cnv.copy()
        d = d.sel(rows=self.train_split.train)
        d['txMid'] = (d.txStart+d.txEnd)/2
        d['data'] = _scale(d.data.astype('float32'))
        d = d.sortby(['chrom', 'txMid'])
        return d
    _playground11.dm_cnv = dm_cnv
    #del self.__lazy__dm_cnv

    @lazy_property
    def gdc_cnv(self):
        d = merge.gdc_cnv.copy()
        d['txMid'] = (d.txStart + d.txEnd) / 2
        d.data.data = d.data.data.rechunk((None, -1))
        d['data'] = _scale(d.data.astype('float32'))
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            d = d.sortby(['chrom', 'txMid'])
        return d
    _playground11.gdc_cnv = gdc_cnv
    #del self.__lazy__gdc_cnv

    @lazy_property
    def dm_cnv_fft(self):
        d = self.dm_cnv.data.assign_coords(arm=self.dm_cnv.arm)
        d = d.drop(['mean', 'var'])
        d = _fft1(d, 10, 'arm', 'cols')
        d = d.drop('arm')
        return d
    _playground11.dm_cnv_fft = dm_cnv_fft
    #del self.__lazy__dm_cnv_fft

    @lazy_property
    def dm_cnv_fft_stats(self):
        d = self.dm_cnv_fft

        d1 = xa.Dataset()
        d1['fft_mean'] = d.fft.mean(axis=0)
        d1['fft_var'] = d.fft.var(axis=0)

        d2 = xa.Dataset()
        d2['fft_resid_mean'] = d.fft_resid.mean(axis=0)
        d2['fft_resid_var'] = d.fft_resid.var(axis=0)

        d = xa.merge([d1, d2])
        return d
    _playground11.dm_cnv_fft_stats = dm_cnv_fft_stats
    #del self.__lazy__dm_cnv_fft_stats

    @lazy_property
    def gdc_cnv_fft(self):
        d = self.gdc_cnv.data.assign_coords(arm=self.gdc_cnv.arm)
        d = d.drop(['mean', 'var'])
        d = _fft1(d, 10, 'arm', 'cols')
        d = d.drop('arm')
        return d
    _playground11.gdc_cnv_fft = gdc_cnv_fft
    #del self.__lazy__gdc_cnv_fft

    @lazy_property
    @cached_property(type=Dir.pickle)
    def dm_cnv_fft_group(self):
        x1 = (self.dm_cnv_fft.fft_resid**2).mean(axis=0)
        x2 = (self.gdc_cnv_fft.fft_resid**2).mean(axis=0)
        x3 = xa.merge([x1.rename('dm'), x2.rename('gdc')])
        x3['arm'] = self.dm_cnv.arm
        x3 = x3.to_dataframe().reset_index()

        x1 = self.dm_cnv_fft.fft_resid.data
        x1 = x1/daa.sqrt((x1**2).sum(axis=0, keepdims=True))
        x1 = x1[:,1:] * x1[:,:-1]
        x1 = x1.sum(axis=0)
        x1 = x1.compute()
        x3['dm_cor'] = np.hstack([x1, [0]])

        x1 = self.gdc_cnv_fft.fft_resid.data
        x1 = x1/daa.sqrt((x1**2).sum(axis=0, keepdims=True))
        x1 = x1[:,1:] * x1[:,:-1]
        x1 = x1.sum(axis=0)
        x1 = x1.compute()
        x3['gdc_cor'] = np.hstack([x1, [0]])

        x1 = x3.arm.to_numpy()
        x1 = np.hstack([True, x1[1:] != x1[:-1]])
        x2 = np.hstack([True, (x3.dm_cor<0.3)[:-1]])
        x3['g1'] = np.cumsum(x1 | x2)

        x3['dm_n'] = x3.groupby('g1').dm.transform(lambda x: sum(x>0.1))
        x3['gdc_n'] = x3.groupby('g1').gdc.transform(lambda x: sum(x>0.1))
        x3['t'] = (x3.gdc>0.1) & (x3.dm>0.1)
        x3['both_n'] = x3.groupby('g1').t.transform(lambda x: sum(x>0.1))

        x4 = x3.query('both_n>0').copy()
        x4['g2'] = x4.g1.astype('category').cat.codes+1
        x4 = x4.groupby('g2').apply(lambda x: x.loc[x.index[x.t][0]:(x.index[x.t][-1]),:])

        return x4.set_index('cols').g2.astype(int)

        x4.query('g2==28 & t==True')

        np.corrcoef(x3.dm, x3.gdc)
        x3['g1'] = x3.g.astype(str)
        px.scatter(
            x3,
            x='dm', y='gdc', color='g1',
            hover_data=['cols']
        ).show()
    _playground11.dm_cnv_fft_group = dm_cnv_fft_group
    #del self.__lazy__dm_cnv_fft_group

    @lazy_property
    def dm_cnv_fft_svd(self):
        s = Path(self.storage.path)/'dm_cnv_fft_svd'
        if not s.exists():
            fft = self.dm_cnv_fft.copy()
            fft['fft_group'] = self.dm_cnv_fft_group
            fft.fft.data = fft.fft.data.persist()
            fft.fft_resid.data = fft.fft_resid.data.persist()
            svd = _svd1(fft, 10, 'cols')
            for i, x in svd.items():
                x.xarray.to_zarr(s/str(i))
        svd = {
            int(i.name):SVD.from_xarray(xa.open_zarr(s/i))
            for i in s.glob('*')
        }
        return svd
    _playground11.dm_cnv_fft_svd = dm_cnv_fft_svd
    # del self.__lazy__dm_cnv_fft_svd

    @lazy_property
    def dm_cnv_fft_svd_rand(self):
        fft = self.dm_cnv_fft
        fft['fft_group'] = self.dm_cnv_fft_group
        return _svd1(fft, 10, 'cols', True)
    _playground11.dm_cnv_fft_svd_rand = dm_cnv_fft_svd_rand
    # del self.__lazy__dm_cnv_fft_svd_rand

    @lazy_property
    @cached_property(type=Dir.pickle)
    def dm_cnv_fft_svd_pc(self):
        m = self
        x1 = m.dm_cnv_fft_svd
        x1 = [x.ve.data for x in x1.values()]
        x1 = daa.hstack(x1)
        x1 = x1.compute()
        x4 = m.dm_cnv_fft_svd_rand
        x4 = [x4[i].ve.data for i in m.dm_cnv_fft_svd.keys()]
        x4 = daa.hstack(x4)
        x4 = x4.compute()
        x2 = pd.DataFrame(dict(
            fft_group_pc = np.hstack([np.repeat(i, x.s.shape[0]) for i, x in m.dm_cnv_fft_svd.items()]).astype(int),
            pc = np.hstack([np.arange(x.s.shape[0]) for _, x in m.dm_cnv_fft_svd.items()]).astype(int),
            n = np.hstack([np.repeat(x.s.shape[0], x.s.shape[0]) for _, x in m.dm_cnv_fft_svd.items()]),
            ve = x1,
            ve_rand = x4
        ))
        x2 = x2.query('n==1 | ve_rand-ve>=1e-4').groupby('fft_group_pc').first()
        return x2.pc+1
    _playground11.dm_cnv_fft_svd_pc = dm_cnv_fft_svd_pc
    # del self.__lazy__dm_cnv_fft_svd_pc

    @property
    def dm_expr1(self):
        x1 = self.dm_cnv_fft_svd[0]
        x1 = x1.u[:,:int(self.dm_cnv_fft_svd_pc[0].item())]
        x6 = self.dm_expr.data
        x6 = x6 - x1[0] @ (x1[0].T @ x6)
        x6 = dmlp.StandardScaler().fit_transform(x6)
        x6 = x6.persist()
        return x6
    _playground11.dm_expr1 = dm_expr1

    @lazy_property
    def dm_expr_svd(self):
        s = Path(self.storage.path)/'dm_expr_svd'
        if not s.exists():
            x6 = self.dm_expr1
            svd = SVD.from_mat(x6)
            svd.xarray.to_zarr(s)
        svd = SVD.from_xarray(xa.open_zarr(s))
        return svd
    _playground11.dm_expr_svd = dm_expr_svd
    # del self.__lazy__dm_expr_svd

    @lazy_property
    def dm_expr_svd_rand(self):
        x6 = self.dm_expr1
        x6.data = daa.apply_along_axis(np.random.permutation, 0, x6.data, shape=(x6.shape[0],), dtype=x6.dtype)
        return SVD.from_mat(x6)
    _playground11.dm_expr_svd_rand = dm_expr_svd_rand
    # del self.__lazy__dm_expr_svd_rand

    @lazy_property
    @cached_property(type=Dir.pickle)
    def dm_expr_svd_pc(self):
        m = self
        x1 =  m.dm_expr_svd.ve.compute()
        x4 = m.dm_expr_svd_rand.ve.compute()
        x2 = pd.DataFrame(dict(
            pc = np.arange(m.dm_expr_svd.s.shape[0]),
            ve = x1,
            ve_rand = x4
        ))
        x2 = x2.query('ve_rand-ve>=1e-4').pc.iloc[0]
        return x2+1
    _playground11.dm_expr_svd_pc = dm_expr_svd_pc
    # del self.__lazy__dm_expr_svd_pc

    @lazy_property
    def crispr_cnv_fit(self):
        s = Path(self.storage.path)/'crispr_cnv_fit'
        if not s.exists():
            x1 = self.crispr.data.sel(rows=self.dm_cnv.rows.values).data.persist()
            x2 = [x.u[:,:int(self.dm_cnv_fft_svd_pc[int(i)].item())].data for i, x in self.dm_cnv_fft_svd.items()]
            x2 = [((x1 - x @ (x.T @ x1))**2).mean(axis=0) for x in x2]
            x2 = daa.stack(x2).persist()
            x2 = x2.rechunk((-1, None))
            x4 = [x.perm.u[:,:int(self.dm_cnv_fft_svd_pc[int(i)].item())].data for i, x in self.dm_cnv_fft_svd.items()]
            x4 = [((x1 - x @ (x.T @ x1))**2).mean(axis=0) for x in x4]
            x4 = daa.stack(x4).persist()
            x4 = x4.rechunk((-1, None))
            x2.to_zarr(str(s/'obs'))
            x4.to_zarr(str(s/'rand'))

        x2 = daa.from_zarr(str(s/'obs'))
        x4 = daa.from_zarr(str(s/'rand'))

        x5 = dmlp.StandardScaler().fit(daa.log10(1-x4+1e-5).T)
        x6 = x5.transform(daa.log10(1-x2+1e-5).T).T.persist()
        m = xa.Dataset()
        m['fft_group'] = ('fft_group', np.arange(x2.shape[0]))
        m['cols'] = self.crispr.cols
        m['rand_mean'] = ('fft_group', x5.mean_)
        m['rand_var'] = ('fft_group', x5.var_)
        m['r2'] = (('fft_group', 'cols'), x6)
        return m
    _playground11.crispr_cnv_fit = crispr_cnv_fit
    # del self.__lazy__crispr_cnv_fit

    @lazy_property
    def crispr_expr_fit(self):
        return _fit(self.crispr.data, self.dm_expr.data, 400)
    _playground11.crispr_expr_fit = crispr_expr_fit
    # del self.__lazy__crispr_expr_fit

    @lazy_property
    def crispr_model(self):
        def _build():
            x1 = (self.crispr_cnv_fit.r2[1:,:]>2).sum(axis=1).to_series()
            x1 = x1.pipe(lambda x: x[x > 0])
            x1 = [0] + list(x1.index)
            x1 = {i: self.dm_cnv_fft_svd[i] for i in x1}
            x1 = {i: x.cut(np.s_[:self.dm_cnv_fft_svd_pc[i].item()]) for i, x in x1.items()}
            x1 = {i: x.inv(0) for i, x in x1.items()}
            x1 = {i: xa.merge([x.us.rename('us'), x.v.rename('v')]) for i, x in x1.items()}
            x1 = {i: x.persist() for i, x in x1.items()}
            x1 = {i: x.assign_coords(pc=['cnv:'+str(i)+':'+str(pc) for pc in x.pc.values]) for i, x in x1.items()}
            x1 = {i: x.assign_coords(group=('pc', np.repeat('cnv:' + str(i), x.pc.shape[0]))) for i, x in x1.items()}
            dm_cnv_fft_svd = x1

            x1 = self.dm_expr_svd
            x1 = x1.cut(np.s_[:self.dm_expr_svd_pc])
            x1 = x1.inv(0)
            x1 = xa.merge([x1.us.rename('us'), x1.v.rename('v')])
            x1 = x1.drop(['mean', 'var'])
            x1 = x1.persist()
            x1 = x1.assign_coords(pc = ['expr:'+str(pc) for pc in x1.pc.values])
            x1 = x1.assign_coords(group=('pc', np.repeat('expr' , x1.pc.shape[0])))
            dm_expr_svd = x1

            crispr = self.crispr.data.copy()
            crispr = crispr.drop(['mean', 'var'])
            crispr.data = crispr.data.persist()
            crispr = crispr.rename(rows='_rows')

            v = [x.v for i, x in dm_cnv_fft_svd.items()]
            v = v + [dm_expr_svd.v]
            v = xa.concat(v, 'pc')
            v = v.rename(rows='_rows', pc='rows')

            x1 = self.crispr_cnv_fit.r2[1:,:]
            x1 = x1.drop(['mean', 'var'])
            x1 = x1.rename(fft_group='group')
            x1 = x1.assign_coords(group=('group', ['cnv:'+str(x) for x in x1.group.values]))
            crispr_cnv_fit_r2 = x1>2

            def model(x5):
                x1 = crispr_cnv_fit_r2.loc[:,x5]
                x1 = x1.sel(group=x1)
                x1 = np.hstack([x1.group.values, ['cnv:0', 'expr']])
                x1 = v.sel(rows=v.group.isin(x1))
                x1 = SVD.from_mat(x1).inv(0).usv
                x1 = x1 @ crispr.loc[:, x5]
                return x1

            x1 = [model(x) for x in crispr.cols.values]

            x2 = [set(x.rows.values) for x in x1]
            x2 = set().union(*x2)
            x2 = pd.Series(range(len(x2)), index=sorted(x2))

            x3 = [x.cols.item() for x in x1]
            x3 = pd.Series(range(len(x3)), index=x3)

            import sparse

            x4 = daa.hstack([x.data for x in x1])
            x4 = sparse.COO(
                [
                    x2[np.hstack([x.rows.values for x in x1])],
                    x3[np.hstack([np.repeat(x.cols.values, x.rows.shape[0]) for x in x1])]
                ],
                x4,
                shape=(len(x2), len(x3))
            )
            x4 = x4.astype('float32')
            x4 = xa.DataArray(
                daa.from_array(x4),
                dims=('pc', 'cols'),
                coords={
                    'pc': x2.index,
                    'cols': x3.index
                }
            )

            x7 = {
                'cnv:' + str(i): x.us.drop('fft_group').rename(cols='rows')
                for i, x in dm_cnv_fft_svd.items()
            }
            x7['expr'] = dm_expr_svd.us.rename(cols='rows')

            x8 = SimpleNamespace()
            x8.u = x4
            x8.vs = x7
            x8.crispr_stats = self.crispr[['mean', 'var']].reset_coords(['mean', 'var'])
            x8.expr_stats = self.dm_expr[['mean', 'var']].reset_coords(['mean', 'var'])
            x8.cnv_stats = self.dm_cnv[['mean', 'var', 'chrom', 'arm', 'txMid']].reset_coords(['mean', 'var'])
            x8.cnv_stats = x8.cnv_stats.merge(self.dm_cnv_fft_stats)

            return x8

        s = Path(self.storage.path)/'crispr_model'
        if not s.exists():
            x8 = _build()
            x1 = x8.u.astype('float16')
            x1.data = daa.from_array(x1.data.compute().todense())
            x1.rename('data').to_dataset().to_zarr(s/'u')
            for i, x in x8.vs.items():
                x1 = x.astype('float16')
                x1['rows'] = x1.rows.astype(str)
                x1.rename('data').to_dataset().to_zarr(s / 'vs' /i)
            x8.crispr_stats.to_zarr(s/'crispr_stats')
            x8.expr_stats.to_zarr(s / 'expr_stats')
            x8.cnv_stats.to_zarr(s / 'cnv_stats')

        x8 = SimpleNamespace()
        x8.u = xa.open_zarr(s/'u').data.astype('float32').rename('u')
        x8.vs = {
            i.name: xa.open_zarr(i).astype('float32').data.rename(i.name)
            for i in (s/'vs').glob('*')
        }
        x8.crispr_stats = xa.open_zarr(s/'crispr_stats')
        x8.expr_stats = xa.open_zarr(s / 'expr_stats')
        x8.cnv_stats = xa.open_zarr(s / 'cnv_stats')
        return x8
    _playground11.crispr_model = crispr_model
    # del self.__lazy__crispr_model

    def crispr_predict(self, expr, cnv):
        model = self.crispr_model

        expr = expr - model.expr_stats['mean']
        expr = expr/np.sqrt(model.expr_stats['var'])
        expr = expr.rename(rows='_rows', cols='rows')
        expr.data = expr.data.rechunk((None, -1))

        cnv = cnv - model.cnv_stats['mean']
        cnv = cnv/np.sqrt(model.cnv_stats['var'])
        cnv = xa.merge([cnv.rename('data'), model.cnv_stats[['arm', 'chrom', 'txMid']]])
        cnv = cnv.sortby(['chrom', 'txMid'])
        cnv = cnv.rename(rows='_rows')
        cnv.data.data = cnv.data.data.rechunk((None, -1))

        d = cnv.data.assign_coords(arm=cnv.arm)
        d = _fft1(d, 10, 'arm', 'cols')
        d = d.drop('arm')
        d['fft'] = d.fft - model.cnv_stats['fft_mean']
        d['fft'] = d.fft/np.sqrt(model.cnv_stats['fft_var'])
        d['fft_resid'] = d.fft_resid - model.cnv_stats['fft_resid_mean']
        d['fft_resid'] = d.fft_resid/np.sqrt(model.cnv_stats['fft_resid_var'])
        cnv = cnv.merge(d)

        expr_v = expr @ model.vs['expr']
        cnv_v_glob = cnv.fft.rename(freq='rows') @ model.vs['cnv:0']
        cnv_v_loc = [
            cnv.fft_resid.rename(cols='rows') @ x
            for i, x in model.vs.items()
            if i not in ['cnv:0', 'expr']
        ]
        crispr = [expr_v, cnv_v_glob] + cnv_v_loc
        crispr = xa.concat(crispr, 'pc')
        crispr.data = crispr.data.rechunk((None, -1))
        crispr = crispr @ model.u
        crispr = crispr.rename(_rows='rows')
        crispr = crispr * np.sqrt(model.crispr_stats['var'])
        crispr = crispr + model.crispr_stats['mean']

        return crispr
    _playground11.crispr_predict = crispr_predict

    @lazy_property
    def gdc_prediction(self):
        s = Path(self.storage.path) / 'gdc_prediction'

        if not s.exists():
            expr = merge.gdc_expr.data.copy().astype('float32')
            cnv = merge.gdc_cnv.data.copy().astype('float32')
            crispr_predict = self.crispr_predict(expr, cnv)
            crispr_predict = crispr_predict.astype('float16')
            crispr_predict['rows'] = crispr_predict.rows.astype(str)
            crispr_predict['cols'] = crispr_predict.cols.astype(str)
            crispr_predict.data = crispr_predict.data.rechunk((1000, -1))
            crispr_predict.rename('data').to_dataset().to_zarr(s)

        crispr_predict = xa.open_zarr(s).data.astype('float32')
        return crispr_predict
    _playground11.gdc_prediction = gdc_prediction
    # del self.__lazy__gdc_prediction

    @lazy_property
    def dm_prediction(self):
        s = Path(self.storage.path) / 'dm_prediction'

        if not s.exists():
            expr = merge.dm_expr.data.copy().astype('float32')
            cnv = merge.dm_cnv.data.copy().astype('float32')
            crispr_predict = self.crispr_predict(expr, cnv)
            crispr_predict = crispr_predict.astype('float16')
            crispr_predict['rows'] = crispr_predict.rows.astype(str)
            crispr_predict['cols'] = crispr_predict.cols.astype(str)
            crispr_predict.data = crispr_predict.data.rechunk((1000, -1))
            crispr_predict.rename('data').to_dataset().to_zarr(s)

        crispr_predict = xa.open_zarr(s).data.astype('float32')
        return crispr_predict
    _playground11.dm_prediction = dm_prediction
    # del self.__lazy__dm_prediction

    @lazy_property
    def crispr_model_score(self):
        x1 = self.dm_prediction
        x1 -= x1.mean(axis=0)
        x1 /= np.sqrt((x1**2).sum(axis=0))

        x2 = merge.crispr.data
        x2 -= x2.mean(axis=0)
        x2 /= np.sqrt((x2**2).sum(axis=0))

        x3 = x1*x2
        x3['train'] = self.train_split.train
        x3 = x3.groupby('train').sum(dim='rows')
        x3 = x3.to_dataframe().reset_index().pivot_table(index='cols', columns='train', values='data')

        return x3
    _playground11.crispr_model_score = crispr_model_score
    # del self.__lazy__crispr_model_score

    @lazy_property
    def crispr_prediction(self):
        x1 = self.dm_prediction.copy()
        x1['row_label'] = merge.dm_expr.CCLE_Name
        x1['source'] = ('rows', np.repeat('dm_prediction', x1.rows.shape[0]))

        x2 = self.gdc_prediction.copy()
        x2['row_label'] = merge.gdc_expr.project_id
        x2['source'] = merge.gdc_expr.project_id

        x3 = merge.crispr.data.copy().astype('float32')
        x3['row_label'] = merge.dm_expr.CCLE_Name
        x3['source'] = ('rows', np.repeat('dm_experiment', x3.rows.shape[0]))

        crispr = xa.concat([x1, x2, x3], 'rows')
        crispr = crispr.assign_coords(merge.crispr[['symbol', 'entrez']])
        return crispr
    _playground11.crispr_prediction = crispr_prediction
    # del self.__lazy__crispr_prediction

    def predict(self, col = None, symbol = None, entrez = None):
        if col is None:
            if entrez is None:
                if symbol is None:
                    raise(ValueError('col or symbol or entrez'))
                else:
                    x5 = self.crispr.cols.sel(cols=self.crispr.symbol == symbol).item()
            else:
                x5 = self.crispr.cols.sel(cols=self.crispr.entrez == entrez).item()
        else:
            x5 = col

        x1 = self.crispr_cnv_fit.r2.loc[1:,x5].to_series().pipe(lambda x: x[x>2]).index
        x1 = [0] + list(x1)
        x1 = [x.u[:,:int(self.dm_cnv_fft_svd_pc[int(i)].item())] for i, x in self.dm_cnv_fft_svd.items() if i in x1]
        x1 = x1 + [self.dm_expr_svd.u[:,:self.dm_expr_svd_pc]]
        x1 = [x.assign_coords(pc=([str(i)+':'+x for x in x.pc.values.astype(str)])) for i, x in enumerate(x1)]
        x1 = xa.concat(x1, 'pc').rename({'pc': 'cols'})
        x1 = SVD.from_mat(x1).persist()
        x2 = self.crispr.data.loc[:,x5].compute()
        x3 = x1.u
        x3 = (x3 @ (x3.T @ x2)).compute()
        return xa.merge([
            x2.rename('obs'),
            x3.rename('pred'),
            (1 - (x2 - x3) ** 2).mean().rename('r2'),
            xa.DataArray(
                x1.u.shape[1] / x1.u.shape[0],
                dims=(),
                coords={'cols': x2.cols},
                name='r2_rand'
            )
        ])
    _playground11.predict = predict
_()


self = playground11

self.crispr_model_score.sort_values(True).tail(20)

crispr = self.crispr_prediction

px.scatter(
    crispr.sel(cols=crispr.symbol=='TP63').squeeze().to_dataframe().sort_values('source'),
    'source', 'data',
    hover_data=['row_label']
).show()




self = playground11
self.dm_cnv_fft_svd_pc


m = playground11
plot_fft_resid(m.dm_cnv.merge(m.dm_cnv_fft).drop('mean'))
plot_fft(m.dm_cnv.merge(m.dm_cnv_fft).drop('mean'))

m = playground11
(m.crispr_cnv_fit.r2[1:,:]>3).sum(axis=1).to_series().sort_values()


m = playground11
x5 = m.predict(symbol='WRN')
plt.plot(x5.obs, x5.pred, '.')
print(x5.r2.round(2).item(), x5.r2_rand.round(2).item())


m = playground11
plt.figure()
plt.gca().plot(m.crispr_cnv_fit.r2[0,:], m.crispr_expr_fit.r2, '.')

plt.figure()
plt.gca().plot(m.crispr_cnv_fit.r2[0,:], m.crispr_cnv_fit.r2[1:,:].max(axis=0), '.')

plt.figure()
plt.gca().plot(m.crispr_expr_fit.r2, m.crispr_cnv_fit.r2[1:,:].mean(axis=0), '.')

plt.figure()
plt.gca().plot(m.crispr_cnv_fit.r2[0,:], m.crispr_cnv_fit.r2[1,:], '.')

pd.DataFrame(dict(
    expr=(m.crispr_expr_fit.r2 > 2).values,
    cnv_glob = (m.crispr_cnv_fit.r2[0,:]>2).values,
    cnv_loc = ((m.crispr_cnv_fit.r2[1:,:]>2).sum(axis=0)>2).values
)).value_counts().sort_index()#.reset_index().pivot(index='loc', columns='glob')


