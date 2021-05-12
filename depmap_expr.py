import pandas as pd
import xarray as xa
import numpy as np
from common.defs import lazy_property
from depmap.depmap import public_21q1 as release

class Expr:
    @property
    def release(self):
        return release


    @lazy_property
    def data(self):
        expr = release.expr

        rows = expr.iloc[:,0].rename('rows')

        samples = release.samples.copy()
        samples = samples.rename(columns={'DepMap_ID': 'rows'})
        samples = samples.query('rows.isin(@rows)')

        cols = pd.Series(expr.columns[1:]).to_frame('cols')
        cols['symbol'] = cols.cols.str.replace(' .*$', '', regex=True)
        cols['entrez'] = cols.cols.str.replace('^.*\(|\)$', '', regex=True).astype(int)

        mat = np.array(expr.iloc[:,1:])

        data = xa.Dataset()
        data['rows'] = ('rows', rows)
        data = data.merge(samples.set_index('rows').to_xarray())
        data = data.merge(cols.set_index('cols').to_xarray())
        data['expr'] = (('rows', 'cols'), mat)

        return data


expr = Expr()