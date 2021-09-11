import taichi as ti
import time

ti.init(arch=ti.gpu)

length = 0.25
originx, originy = 0.45, 0.1
N = 32
dt = 1e-5
dx = 1 / N
rho = 4e1
NF = 2 * N ** 2  # number of faces
NV = (N + 1) ** 2  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.32
gravity = -10  # -40
# confining = 2e3
confining = 0.
damping = 12.5
# print(ti.exp(-dt*damping)**30)

pos = ti.Vector.field(2, float, NV, needs_grad=True)
vel = ti.Vector.field(2, float, NV)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)  # every single B[i] belongs to a faces
F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)  # gradient of the displacement
V = ti.field(float, NF)
phi = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
U = ti.field(float, (), needs_grad=True)  # total potential energy
# boundary
uMaskTop = ti.Vector.field(2, int, NV)
uLoadMap = ti.Vector.field(2, float, NV)
uMaskBottom = ti.Vector.field(2, int, NV)
uMaskFixed = ti.Vector.field(2, int, NV)
t = 0.
du = -1e-9
pressureBoundary = ti.Vector.field(2, float, NV)

@ti.kernel
def update_U():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        V[i] = abs((a - c).cross(b - c))
        # print(i, V[i])
        D_i = ti.Matrix.cols([a - c, b - c])  # what is D_i
        F[i] = D_i @ B[i]
    for i in range(NF):
        F_i = F[i]
        log_J_i = ti.log(F_i.determinant())
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2) - mu * log_J_i + lam / 2 * log_J_i ** 2
        # phi_i -= mu * log_J_i
        # phi_i += lam / 2 * log_J_i**2
        phi[i] = phi_i
        U[None] += V[i] * phi_i


@ti.kernel
def advance(iter: ti.i8):
    for i in range(NV):
        acc = (-pos.grad[i] / (rho * dx ** 2))
        # vel[i] += dt * (acc + gravity)
        vel[i] += dt * (acc + pressureBoundary[i])
        vel[i] *= ti.exp(-dt * damping)
    for i in range(NV):
        # ball boundary condition:
        # disp = pos[i] - ball_pos
        # disp2 = disp.norm_sqr()
        # if disp2 <= ball_radius**2:
        #     NoV = vel[i].dot(disp)
        #     if NoV < 0:
        #         vel[i] -= NoV * disp / disp2  # remove the normal component of the velocity vector
        # rect boundary condition:
        cond = (pos[i] < 0 and vel[i] < 0) or (pos[i] > 1 and vel[i] > 0)
        for j in ti.static(range(pos.n)):
            if cond[j]:
                vel[i][j] = 0
        # add velocity constraint to the bottom and top nodes
        if iter-1<0.:
            pos[i] = pos[i] + \
                     dt * vel[i] * (1.0 - (uMaskTop[i] + uMaskBottom[i])) * (1 - uMaskFixed[i])
            # print(0)
        else:
            pos[i][1] = (pos[i][1]-originy)*0.999+originy
            # pos[i] += du * uLoadMap[i]


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) / N * length + ti.Vector([originx, originy])
        vel[k] = ti.Vector([0, 0])
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()


@ti.kernel
def init_mesh():
    """
        triangle element used in this calculation
    :return:
    """
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


@ti.kernel
def init_pressureBoundary():
    minx = originx
    maxx = originx + length
    for i in range(NV):
        left = (pos[i][0] == minx)
        right = (pos[i][0] == maxx)
        if left:
            pressureBoundary[i] = [confining, gravity]
        elif right:
            pressureBoundary[i] = [-confining, gravity]
        else:
            pressureBoundary[i] = [0, gravity]


@ti.kernel
def init_uMask():
    miny = originy
    maxy = length + originy
    nTop, nBottom = 0, 0
    for i in range(NV):
        bottom = (pos[i][1] == miny)
        top = (pos[i][1] == maxy)
        fix = (pos[i][1] == miny) and (abs(pos[i][0] - originx - length / 2.0) < length / 15)
        if bottom:
            nBottom += 1
            uMaskBottom[i] = [0, 1]
        elif top:
            nTop += 1
            uMaskTop[i] = [0, 1]
        if fix:
            uMaskFixed[i] = [1, 1]
        uLoadMap[i] = [0, (pos[i][1]-originy)/length]



init_mesh()
init_pos()
init_pressureBoundary()
init_uMask()
gui = ti.GUI('FEM99')
nstep = 0
while gui.running:
    # for e in gui.get_events():
    #     if e.key == gui.ESCAPE:
    #         gui.running = False
    #     elif e.key == 'r':
    #         init_pos()
    for i in range(30):
        with ti.Tape(loss=U):
            update_U()
        advance(0)  # iteration scope
        # print(0)
    advance(1)  # add displacement boundary condition
    time.sleep(0.1)
    t += dt
    nstep += 1
    # print(nstep)
    gui.circles(pos.to_numpy(), radius=2, color=0xffaa33)
    # gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)  # plot the rigid circle
    gui.show()
