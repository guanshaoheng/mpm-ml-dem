import numpy as np
from matplotlib import pyplot as plt


def kernel(x1, x2):
    k = np.linalg.norm(x1-x2)
    return k


def kernelMatrix(X1, X2):
    n1, n2 = len(X1), len(X2)
    K = np.zeros(shape=(n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel(X1[i], X2[j])
    return K

# sample nodes
n_numbers = 100
x = np.random.uniform(0, 2.*np.pi, n_numbers)
y = np.sin(x)+np.random.rand(n_numbers)*0.1
x_hat = np.concatenate((np.ones([n_numbers, 1]), x.reshape(-1, 1)), axis=1)

# node in the function
nn_numbers = 50
xx = np.linspace(0, np.pi, nn_numbers)
xx_hat = np.concatenate((np.ones([nn_numbers, 1]), xx.reshape(-1, 1)), axis=1)

kk = 1+kernelMatrix(x, x)
alpha = np.linalg.inv(kk)@y.reshape(-1, 1)
K = kernelMatrix(x_hat, x_hat)
yy = K@alpha
err_average = np.mean(yy-y)

plt.scatter(x, y, marker='o', color='b', s=1.5, alpha=0.5)
plt.scatter(x, yy, marker='o', c='r', s=5)

plt.show()
