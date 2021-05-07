if __name__ != '__main__':
    raise ValueError('can only run as main')

from pathlib import Path
from common.dir import Dir
from helpers import config
import ae_expr as ae
import sys

config.exec()

storage = Path('output/expr/ae-fit')
storage.mkdir(parents=True, exist_ok=True)

models = {
    'model1': ae.model1,
    'model2': ae.model2
}

model_name = sys.argv[1]
epochs = int(sys.argv[2])

model = models[model_name]()
model.data =  ae.data2()
model.kwargs = {'cp_callback': {'save_freq': 12, 'verbose': 1}}
model.data.storage = Dir(storage/'data')
model.storage = Path(model.data.storage.path)/model_name
model.fit(epochs=epochs)

