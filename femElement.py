import numpy as np


class TriangleElement:
    def __init__(self, x1, y1, x2, y2, x3, y3):
        self.n1 = np.array([x1, y1])
        self.n2 = np.array([x2, y2])
        self.n3 = np.array([x3, y3])
        self.area = .5*abs(np.cross(self.n1-self.n2, self.n1-self.n3))

    def aa(self, N):
        a1 = N[1][0]*N[2][1]-N[2][0]*N[1][1]
        a2 = N[1][0]*N[2][1]-N[2][0]*N[1][1]
        a3 = N[1][0]*N[2][1]-N[2][0]*N[1][1]
        b = N[1][1]-N[0][1]
        c = N[2][0]-N[1][0]
