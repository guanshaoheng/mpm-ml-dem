import numpy as np
import matplotlib.pyplot as plt

tempList = np.linspace(0, 1, 10)

X = []
for i in tempList:
    for j in tempList:
        X.append([i, j])
X = np.array(X)

theta = 30/180*np.pi
a, b = np.sin(theta), np.cos(theta)
x = np.concatenate((0.8*X[:, 0:1]+0.01*X[:, 1:2]*X[:, 1:2],
                   0.01*X[:, 0:1]+0.9*X[:, 1:2]), axis=1)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(x[:, 0], x[:, 1])
ax.set_aspect('equal')
plt.show()