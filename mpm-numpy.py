import numpy as np
import matplotlib.pyplot as plt
from math import floor

n_partiles = 90
n_grid = 20
dx, inv_dx = 1. / n_grid, float(n_grid)
dt = 1e-4
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
particles = np.zeros(shape=(n_partiles, 2), dtype=np.float32)
v = np.zeros(shape=(n_partiles, 2), dtype=np.float32)
Jp = np.ones(shape=[n_partiles])
material = np.zeros(shape=[n_partiles]) + 2
F = np.zeros(shape=(n_partiles, 2, 2), dtype=np.float32)
C = np.zeros(shape=(n_partiles, 2, 2), dtype=np.float32)
grid_v = np.zeros(shape=(n_grid, n_grid, 2), dtype=np.float32)
grid_m = np.zeros(shape=(n_grid, n_grid), dtype=np.float32)
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
        (1 + nu) * (1 - 2 * nu))  # Lame parameters


def load():
    for i in range(n_grid):
        for j in range(n_grid):
            grid_v[i, j] = np.array([0., 0.], np.float32)
            grid_m[i, j] = 0.

    # particle 2 grid
    for i, p in enumerate(particles):
        base = np.floor(p * inv_dx - 0.5)
        # base = np.array([int(p[i]*inv_dx-0.5) for i in range(len(p))])
        fx = p * inv_dx - base
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1) ** 2,
            0.5 * (fx - 0.5) ** 2
        ]
        F[i] = (np.diag([1, 1]) + dt * C[i]) @ F[i]

        h = np.exp(10 * (1.0 - Jp[i]))  # Hardening coefficient: snow gets harder when compressed
        if material[i] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[i] == 0:  # liquid
            mu = 0.0

        U, sig, V = np.linalg.svd(F[i])
        sig = np.diag(sig)
        J = 1.0

        for d in range(2):
            new_sig = sig[d, d]
            if material[i] == 2:
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[i] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        J = 0.0
        for d in range(2):
            J+=sig[d, d]/2.

        if material[i] == 0:
            F[i] = np.diag([1, 1]) * np.sqrt(J)
        if material[i] == 2:
            F[i] = U @ sig @ V.T
        # stress = 2 * mu * (F[i] - U @ V.T) @ F[i].T + np.diag([1, 1]) * la * J * (J - 1)
        stress = 2 * mu * (F[i]-np.diag([1, 1])) + np.diag([1, 1]) * la * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[i]

        for i in range(3):
            for j in range(3):  # Loop over 3x3 grid node neighborhood
                offset = np.array([i, j])
                dpos = (offset - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_coord = [int(base[0]) + i, int(base[1]) + j]
                # grid_coord = base + offset
                grid_v[grid_coord[0], grid_coord[1]] += weight * (p_mass * v[i] + affine @ dpos)
                grid_m[grid_coord[0], grid_coord[1]] += weight * p_mass

    # grid solution
    for i in range(n_grid):
        for j in range(n_grid):
            if grid_m[i, j] > 0:
                grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
                grid_v[i, j][1] -= dt * 50  # gravity
                if i < 3 and grid_v[i, j][0] < 0:
                    grid_v[i, j][0] = 0  # Boundary conditions
                if i > n_grid - 3 and grid_v[i, j][0] > 0:
                    grid_v[i, j][0] = 0
                if j < 3 and grid_v[i, j][1] < 0:
                    grid_v[i, j][1] = 0
                if j > n_grid - 3 and grid_v[i, j][1] > 0:
                    grid_v[i, j][1] = 0

    for p in range(n_partiles):  # grid to particle (G2P)
        base = np.floor(particles[p] * inv_dx - 0.5)
        fx = particles[p] * inv_dx - base
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2
        ]
        new_v = np.array([0.]*2)
        new_C = np.diag([0., 0.])
        for i in range(3):
            for j in range(3):  # loop over 3x3 grid node neighborhood
                dpos = np.array([i, j]) - fx
                # grid_coord = base + offset
                # grid_v[int(grid_coord[0]), int(grid_coord[1])
                grid_coord = [int(base[0])+i, int(base[1])+j]
                g_v = grid_v[grid_coord[0], grid_coord[1]]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * np.outer(g_v, dpos)
        v[p], C[p] = new_v, new_C
        particles[p] += dt * v[p]  # advection


def initialize():
    for p in range(len(particles)):
        particles[p] = np.array([
            np.random.rand() * 0.2 + 0.3,  # x, y coords
            np.random.rand() * 0.2 + 0.1
        ])
        F[p] = np.diag([1, 1])


# simulation
initialize()


fig, axes = plt.subplots(1, 1, figsize=(7, 6))
# plt.ion()  # 开启interactive mode 成功的关键函数
for i in range(int(1e5)):
    # ax0, ax1, ax2, ax3 = axes.ravel()
    if i % 100 == 0:
        print(i)
        plt.scatter(particles[:, 0], particles[:, 1])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.tight_layout()
        plt.savefig('./load/%d.png' % i)
        plt.close()
    # plt.draw()
    load()



