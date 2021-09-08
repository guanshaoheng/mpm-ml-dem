import numpy as np
from matplotlib import pyplot as plt

"""
y= wx

w = x.T@y/(x.T@xx)

moving least square
"""

x = np.random.randn(10000)
y = 2*x+np.random.randn(10000)*1.0

w = x.T@y/(x.T@x)
print(w)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y)
ax.plot(np.linspace(min(x), max(x), 10), w*np.linspace(min(x), max(x), 10), c='k')
plt.savefig('regression.svg')
