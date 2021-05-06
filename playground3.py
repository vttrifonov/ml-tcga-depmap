from pathlib import Path
from common.dir import Dir
from helpers import config
import ae_expr as ae

config.exec()

storage = Path(config.cache/'playground2')

model1 = ae.model1()
model1.data =  ae.data1()
model1.data.storage = Dir(storage/'data')
model1.storage = Path(model1.data.storage.path)/'model1'
model1.fit(epochs=1)

model2 = ae.model2()
model2.data =  ae.data1()
model2.data.storage = Dir(storage/'data')
model2.storage = Path(model2.data.storage.path)/'model2'
model2.fit(epochs=1)
