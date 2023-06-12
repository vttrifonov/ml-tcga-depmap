import pandas as pd
from .depmap_crispr import crispr as depmap_crispr
from .depmap_expr import expr as depmap_expr
from .depmap_cnv import cnv as depmap_cnv
from .gdc_expr import expr as gdc_expr
from .gdc_cnv import cnv as gdc_cnv
from .common.defs import lazy_property

def order_set(s, x):
    s = list(s)
    s = pd.Series(range(len(x)), index=x)[s].sort_values().index
    return list(s)

class _merge:
    @property
    def crispr1(self):
        return depmap_crispr.mat3

    @property
    def dm_expr1(self):
        return depmap_expr.mat3

    @property
    def dm_cnv1(self):
        return depmap_cnv.mat3

    @property
    def gdc_expr1(self):
        return gdc_expr.mat3

    @property
    def gdc_cnv1(self):
        return gdc_cnv.mat3

    @lazy_property
    def dm_rows(self):
        crispr = self.crispr1
        dm_cnv = self.dm_cnv1
        dm_expr = self.dm_expr1
        depmap_rows = set(crispr.rows.values) & set(dm_expr.rows.values) & set(dm_cnv.rows.values)
        depmap_rows = order_set(depmap_rows, crispr.rows)
        return depmap_rows

    @lazy_property
    def gdc_rows(self):
        gdc_expr = self.gdc_expr1
        gdc_cnv = self.gdc_cnv1
        gdc_rows = set(gdc_expr.rows.values) & set(gdc_cnv.rows.values)
        gdc_rows = order_set(gdc_rows, gdc_expr.rows)
        return gdc_rows

    @lazy_property
    def expr_cols(self):
        dm_expr = self.dm_expr1
        gdc_expr = self.gdc_expr1
        expr_cols = set(dm_expr.cols.values) & set(gdc_expr.cols.values)
        expr_cols = order_set(expr_cols, gdc_expr.cols)
        return expr_cols

    @lazy_property
    def cnv_cols(self):
        dm_cnv = self.dm_cnv1
        gdc_cnv = self.gdc_cnv1
        cnv_cols = set(dm_cnv.cols.values) & set(gdc_cnv.cols.values)
        cnv_cols = order_set(cnv_cols, gdc_cnv.cols)
        return cnv_cols

    @lazy_property
    def crispr(self):
        return self.crispr1.sel(rows=self.dm_rows)

    @lazy_property
    def dm_cnv(self):
        return self.dm_cnv1.sel(rows=self.dm_rows, cols=self.cnv_cols)

    @lazy_property
    def dm_expr(self):
        return self.dm_expr1.sel(rows=self.dm_rows, cols=self.expr_cols)

    @lazy_property
    def gdc_cnv(self):
        return self.gdc_cnv1.sel(rows=self.gdc_rows, cols=self.cnv_cols)

    @lazy_property
    def gdc_expr(self):
        return self.gdc_expr1.sel(rows=self.gdc_rows, cols=self.expr_cols)

merge = _merge()