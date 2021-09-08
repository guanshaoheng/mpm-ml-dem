import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


datas = pd.read_csv('./data.out')
da = datas.values

plt.plot(da[:, 0], da[:, 1])
plt.show()