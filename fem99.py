import taichi as ti

ti.init(arch=ti.gpu)

N = 32
dt = 1e-4
dx = 1 / N
rho = 4e1
NF = 2 * N**2  # number of faces
NV = (N + 1)**2  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
C1, D1 = mu/2., lam/2.
third = 1/3.
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.32
gravity = ti.Vector([0, -40])
eye = ti.Matrix([[1., 0.], [0., 1.]])
damping = 12.5
print(ti.exp(-dt*damping))

pos = ti.Vector.field(2, float, NV, needs_grad=True)
vel = ti.Vector.field(2, float, NV)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)  # every single B[i] belongs to a faces
F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)  # gradient of the displacement
V = ti.field(float, NF)
phi = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
U = ti.field(float, (), needs_grad=True)  # total potential energy


@ti.kernel
def update_U():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        V[i] = abs((a - c).cross(b - c))  # area of the element (face)
        D_i = ti.Matrix.cols([a - c, b - c])  # what is D_i
        F[i] = D_i @ B[i]
    for i in range(NF):
        F_i = F[i]
        # neo-hookean
        log_J_i = ti.log(F_i.determinant())
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)-mu*log_J_i + lam/2*log_J_i**2
        # phi_i -= mu * log_J_i
        # phi_i += lam / 2 * log_J_i**2
        phi[i] = phi_i
        U[None] += V[i] * phi_i

        # ELASTIC
        # dsp_gradient = F_i - eye
        # epsilon_i = 0.5*(dsp_gradient+dsp_gradient.transpose())
        # tr_epsilon = epsilon_i.trace()
        # stress_i = lam*eye*tr_epsilon+2*mu*epsilon_i
        # energy_i = stress_i * epsilon_i
        # phi_i = energy_i[0, 0]+energy_i[1, 1]+energy_i[1, 0]*2.
        # U[None] += V[i] * phi_i
        # print(i, stress_i[0, 0], stress_i[0, 1], stress_i[1, 0], stress_i[1, 1], '\t',
        #       epsilon_i[0, 0], epsilon_i[0, 1], epsilon_i[1, 0], epsilon_i[1, 1], '\t', phi_i)


@ti.kernel
def advance():
    for i in range(NV):
        acc = -pos.grad[i] / (rho * dx**2)
        vel[i] += dt * (acc + gravity)
        vel[i] *= ti.exp(-dt * damping)
    for i in range(NV):
        # ball boundary condition:
        disp = pos[i] - ball_pos
        disp2 = disp.norm_sqr()
        if disp2 <= ball_radius**2:
            NoV = vel[i].dot(disp)
            if NoV < 0:
                vel[i] -= NoV * disp / disp2  # remove the normal component of the velocity vector
        # rect boundary condition:
        cond = (pos[i] < 0 and vel[i] < 0) or (pos[i] > 1 and vel[i] > 0)
        for j in ti.static(range(pos.n)):
            if cond[j]:
                vel[i][j] = 0
        pos[i] += dt * vel[i]


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) / N * 0.25 + ti.Vector([0.45, 0.45])
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


init_mesh()
init_pos()
gui = ti.GUI('FEM99')
while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'r':
            init_pos()
    for i in range(30):
        with ti.Tape(loss=U):
            update_U()
        advance()
    # update_U()
    # advance()
    gui.circles(pos.to_numpy(), radius=2, color=0xffaa33)
    gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
    gui.show()
