import bifeatureanalysis as ba
import os
import numpy as np
import pandas as pd

y = [1, 2, 3, 4]
y_predicted = [1.5, 2.5, 3.5, 4.5]

# y = np.asarray(y)
# h = y.size
print('hello world')
result = ba.hubber_loss(y, y_predicted)
print(result)
