import numpy as np
from matplotlib import pyplot as plt


x1, x2 = 0, 1.0
u1 = 0.0
u2 = 1.0
x = np.linspace(0.01, 1.0-0.01, 100)
z = np.linspace(0.01, 1.0-0.01, 100).reshape(-1, 1)

N1, N2 = (x2-x)/(x2-x1), (x-x1)/(x2-x1)

ux = N1*u1+N2*u2
uz = (z+x2-2*x)*u1+(2*x-x1-z)*u2

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(uz[:, 49])
plt.show()
