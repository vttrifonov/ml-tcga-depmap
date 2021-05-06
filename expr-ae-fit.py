if __name__ != '__main__':
    raise ValueError('can only run as main')

from pathlib import Path
from common.dir import Dir
from helpers import config
import ae_expr as ae
import sys

config.exec()

storage = Path('results/expr/ae-fit')
storage.mkdir(parents=True)

models = {
    'model1': ae.model1,
    'model2': ae.model2
}

model_name = sys.argv[1]
epochs = int(sys.argv[2])

model = models[model_name]()
model.data =  ae.data1()
model.data.storage = Dir(storage/'data')
model.storage = Path(model.data.storage.path)/model_name
model.fit(epochs=epochs)
