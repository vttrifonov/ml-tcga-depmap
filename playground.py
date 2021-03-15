import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common.defs import pipe, lapply
import seaborn as sb
import more_itertools as mit
import ae
import analysis1
importlib.reload(ae)
importlib.reload(analysis1)
import ae
import analysis1

def plot_roc(obs, pred):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(obs, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2, label=f'AUC = {roc_auc:.02f}'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.show()

def roc(obs, pred):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(obs, pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def m_test_aucs(m):
    return m.trrange(m.data.test.shape[0]) |pipe| \
        lapply(lambda i: roc(m.data.test[i, :], m.decoded[i, :])) |pipe|\
        lapply(lambda x: x[2]) |pipe| list

def m_train_aucs(m):
    return m.train_decoded |pipe|\
        lapply(lambda x: roc(x[0][0, :], x[1][0,: ])) |pipe|\
        lapply(lambda x: x[2]) |pipe| list

a = analysis1.analysis1

cases = a.snv.cases
cases = cases[cases.project_id == 'TCGA-COAD'].case_id.reset_index(drop=True)
#cases = cases.case_id

d = a.snv_data1(cases)
d.m = {}

d.m['pca'] = d.fit(ae.PCA(100))
d.m['pca'].ae.model.fit(mit.nth(d.train.batch(sum(d.select)), 0)[0])

d.m['ae1'] = d.fit(ae.AE(len(d.mat.colnames), 100, 'linear', 'linear', 'adam', 'mse'))
d.m["ae1"].ae.model.fit(
    d.train.batch(sum(d.select)).repeat(),
    epochs=100, steps_per_epoch=1
)

d.m['ae2'] = d.fit(ae.AE(len(d.mat.colnames), 100, 'relu', 'sigmoid', 'adam', 'binary_crossentropy'))
d.m["ae2"].ae.model.fit(
    d.train.batch(sum(d.select)).repeat(),
    epochs=100, steps_per_epoch=1
)

d.m['ae3'] = d.fit(ae.AE1(len(d.mat.colnames), [200, 100], [200]))
d.m['ae3'].fit(
    d.train.batch(sum(d.select)).repeat(),
    epochs=100, steps_per_epoch=1
)

d.m['ae4'] = d.fit(ae.AE1(len(d.mat.colnames), [(200, 0.01, 1), 100], [200]))
d.m['ae4'].fit(
    d.train.batch(sum(d.select)).repeat(),
    epochs=100, steps_per_epoch=1
)

d.m |pipe|\
    (lambda x: ((k, v) for k, v in x.items())) |pipe|\
    lapply(lambda x: pd.DataFrame({"m": x[0], "x": m_train_aucs(x[1])})) |pipe|\
    (lambda x: (pd.concat(x),)) |pipe|\
    (lambda x: sb.kdeplot(x='x', hue='m', data=x[0]))

z = pd.DataFrame({
    'id': range(d.test.shape[0]),
    'x': m_test_aucs(d.m['pca']),
    'y': m_test_aucs(d.m['ae1'])
})

sb.scatterplot(x=d.m['pca'].decoded[51], y=d.m['ae1'].decoded[51])

m1 = d.m['pca'].ae.model.components_
m2 = d.m['ae1'].ae.model.layers[2].trainable_weights[0].numpy()

plt.hist(np.diag(m2 @ m2.T))

from sklearn.decomposition import PCA

x1 = PCA().fit(m1)
x2 = PCA().fit(m2)

plt.imshow(x1.components_)
plt.hist((x1@m1.T).flatten(), 100)

(d.m['ae4'], 3) |pipe| (lambda x: sb.boxplot(x[0].data.test[x[1], :], x[0].decoded[x[1], :]))

(d.m['ae4'], 3) |pipe| (lambda x: plot_roc(x[0].data.test[x[1], :], x[0].decoded[x[1], :]))

d.m['ae4'] |pipe| (lambda x: m_train_aucs(x)) |pipe| (lambda x: np.quantile(x, [0, 1]))
