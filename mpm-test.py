import numpy as np

## parameters
l = 1.0
dt = 0.01
time = 1

## node
n_node = 2
x_node = np.linspace(0, l, n_node)
v_node = np.zeros(n_node)
m_node = np.zeros(n_node)
momentum_node = np.zeros(n_node)


## particle
n_p = 1
x_p = .5*l
v_p = 1.0
m_p = 1.0
momentum_p = v_p*m_p



## simulation
t = 0.
xp_hist = []
while t<tiem:
    # particle 2 grid (mass, velocity)

    # grid momentum update

    # grid to particle (velocity, coordinate)

    # grid velocity and strain on the particles

    # stress update

    

