import numpy as np
import matplotlib.pyplot as plt
from math import floor


x = np.linspace(0, 1.0/128., int(1e6))


def fx(temp):
    base = floor(temp*128.-0.5)
    f = temp*128.-base
    return f

def wx(temp):
    # w1 = [0.5 * (1.5 - temp) ** 2, 0.75 - (temp - 1.0) ** 2, 0.5 * (temp - 0.5) ** 2]
    w1 = 0.5 * (1.5 - temp) ** 2
    w2 = 0.75 - (temp - 1.0) ** 2
    w3 = 0.5 * (temp - 0.5) ** 2
    summ = w1+w2+w3
    return w1, w2, w3, summ

fig = plt.figure()
ax = fig.add_subplot(111)
ffx = [fx(i) for i in x]
ww1 = [wx(i)[0] for i in ffx]
ww2 = [wx(i)[1] for i in ffx]
ww3 = [wx(i)[2] for i in ffx]
summ = [wx(i)[3] for i in ffx]
# summ = [ww1[i]+ww2[i]+ww3[3] for i in range(len(ww3))]
ax.plot(x, ffx, label='fx')
ax.plot(x, ww1, label='w1')
ax.plot(x, ww2, label='w2')
ax.plot(x, ww3, label='w3')
ax.plot(x, summ, label='sum')
plt.legend()
plt.show()
