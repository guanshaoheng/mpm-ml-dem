import taichi as ti
ti.init()

"""
    This example is used to get the minimum of the y=sin(x)
    
    ti.Tape(loss=y) is used to get the differentiation of the dy/dx
"""

n = 1
x = ti.field(float, n, needs_grad=True)
y = ti.field(float, (), needs_grad=True)


@ti.kernel
def compute_y():
    for i in range(n):
        y[None] += ti.sin(x[i])


@ti.kernel
def advance():
    for i in x:
        x[i] -= x.grad[i]*1e-3


@ti.kernel
def init():
    for i in x:
        x[i] = ti.random()


init()
gui = ti.GUI('tapeTest')
nn = 0
while gui.running:
    for i in range(10):
        with ti.Tape(loss=y):
            compute_y()
        advance()
    nn += 1
    print('dy/dx =%.2f at x=%.2f y=sin(x)=%.2f %d' % (x.grad[0], x[0], y[None], nn))
