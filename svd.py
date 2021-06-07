import xarray as xa
import dask.array as daa
import numpy as np
from common.defs import lazy_property

def _perm(x):
    return x[np.random.permutation(x.shape[0]), :]

class SVD:
    def __init__(self, u, s, v):
        self.u = u
        self.s = s
        self.v = v

    @staticmethod
    def from_mat(mat, n = None, solver = 'full'):
        if n is None:
            n = min(*mat.shape)

        if solver == 'full':
            _svd = daa.linalg.svd(mat.data)
        elif solver == 'rand':
            _svd = daa.linalg.svd_compressed(mat.data, n)
        else:
            raise ValueError('unknown solver')

        _svd = (_svd[0][:,:n], _svd[1][:n], _svd[2][:n,:].T)

        svd = xa.Dataset()
        svd['u'] = ((mat.dims[0], 'pc'), _svd[0])
        svd['s'] = ('pc', _svd[1])
        svd['v'] = ((mat.dims[1], 'pc'), _svd[2])
        svd['pc'] = np.arange(n)
        svd = svd.merge(mat.coords)

        return SVD(svd.u, svd.s, svd.v)

    def cut(self, n=None):
        if n is None:
            n = np.s_[:]
        return SVD(self.u[:, n], self.s[n], self.v[:, n])

    @property
    def us(self):
        return self.u * self.s

    @property
    def vs(self):
        return self.v * self.s

    @property
    def usv(self):
        return self.us @ self.v.T

    @property
    def perm(self):
        u = _perm(self.u)
        u[u.dims[0]] = self.u[u.dims[0]]
        return SVD(u, self.s, self.v)

    def inv(self, l = 0):
        return SVD(self.v, self.s/(l + self.s**2), self.u)

    @property
    def T(self):
        return SVD(self.v, self.s, self.u)

    def lmult(self, x):
        return SVD(x @ self.u, self.s, self.v)

    def rmult(self, x):
        return SVD(self.u, self.s, x.T @ self.v)

    def persist(self):
        return SVD(
            self.u.persist(),
            self.s.persist(),
            self.v.persist()
        )

    @staticmethod
    def from_xarray(x):
        return SVD(x.u, x.s, x.v)

    @property
    def lsvd(self):
        svd = SVD.from_mat(self.us)
        svd.v = self.v @ svd.v
        return svd

    @property
    def xarray(self):
        return xa.merge([self.u.rename('u'), self.s.rename('s'), self.v.rename('v')])

    @lazy_property
    def ve(self):
        ve = self.s**2
        ve = ve / ve.sum()
        return ve.rename('ve')