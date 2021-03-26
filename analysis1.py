import numpy as np
import ae
from snv import snv
from expr import expr

class Analysis1:
    class SNVData(ae.Data):
        def __init__(self, cases, genes):
            if cases is None:
                cases = snv.full_mat.rownames
            if genes is None:
                genes = snv.full_mat.colnames

            self.mat = snv.mat(genes)

            select = np.random.random(len(cases))<0.7
            self.cases = cases
            self.select = select
            self.train = self.mat.tensor(cases[select])
            self.test = self.mat.tensor(cases[~select])

    def snv_data(self, cases = None, genes = None):
        return self.SNVData(cases, genes)

    class ExprData(ae.Data):
        def __init__(self, files, genes):
            if files is None:
                files = expr.full_mat.rownames
            if genes is None:
                genes = expr.full_mat.colnames

            self.mat = expr.mat(genes)

            select = np.random.random(len(files))<0.7
            self.files = files
            self.select = select
            self.train = self.mat.tensor(files[select])
            self.test = self.mat.tensor(files[~select])

    def expr_data(self, files = None, genes = None):
        return self.ExprData(files, genes)

analysis1 = Analysis1()

