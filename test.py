import numpy as np
import matplotlib.pyplot as plt
from math import floor
import taichi as ti

ti.init(arch=ti.gpu)


@ti.kernel
def foo():
    a = ti.Vector([0, 2])
    b = ti.Vector([1, 0])
    c = ti.Matrix.cols([a, b])
    area = abs(a.cross(b))*0.5
    B = c.inverse()
    aa = ti.Vector([0.1, 2])
    bb = ti.Vector([1, 0.01])
    cc = ti.Matrix.cols([aa, bb])
    F = cc@B
    print('a', a, 'c', c, 'area:', area, 'shape', B, 'gradient', F)


foo()
