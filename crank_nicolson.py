import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from util import *

# Grid setup
Ny, Nx = 129, 129
h = 2
N = Nx * Ny

# Parameters setup
Du = 1.0
Dv = 0.5
ff = 0.055
kk = 0.062

# Time steps
dt = 1
end_t = 20
time_steps = int(end_t / dt)
print(time_steps)
c = dt / (h * h)

# Construct the transform matrix of U and V
# L = Laplace_matrix(Ny, Nx)
L = Laplace_matrix_sparse(Ny, Nx)
Tu = sparse.eye(N) - 0.5 * Du * c * L
Tv = sparse.eye(N) - 0.5 * Dv * c * L

# Factorize Tu, Tv
solve_Tu = linalg.factorized(Tu.tocsc())
solve_Tv = linalg.factorized(Tv.tocsc())

# Initial condition
U_init = np.zeros((Ny, Nx))
V_init = np.zeros((Ny, Nx))
U_init[:, :] = 1
V_init[51:60, 51:70] = 1
V_init[61:80, 71:80] = 1

U1d = np.zeros((time_steps, N))
V1d = np.zeros((time_steps, N))
U1d[0] = np.copy(U_init.flatten())
V1d[0] = np.copy(V_init.flatten())

# CN Method
for i in range(1, time_steps):
    bU = (0.5 * Du * c) * L.dot(U1d[i - 1]) \
        + np.multiply((1 - dt * np.square(V1d[i - 1]) - dt * ff), U1d[i - 1]) + dt * ff
    U1d[i] = solve_Tu(bU)

    bV = (0.5 * Dv * c) * L.dot(V1d[i - 1]) + (1 - dt * kk - dt * ff) * V1d[i - 1] \
        + dt * np.multiply(U1d[i - 1], np.square(V1d[i - 1]))
    V1d[i] = solve_Tv(bV)

U2d = U1d.reshape((time_steps, Ny, Nx))
V2d = V1d.reshape((time_steps, Ny, Nx))

np.save('../ndarr_UV/U_cn', U2d)
np.save('../ndarr_UV/V_cn', V2d)

U_cn = np.load('../ndarr_UV/U_cn.npy')
V_cn = np.load('../ndarr_UV/V_cn.npy')

plot_pattern(U_cn, V_cn, time_steps - 1, time_steps - 1, h, dt)

# ani = animate_pattern(U_cn, V_cn, h, dt, Nsteps=time_steps, Nout=2)
# ani.save('../ani/pattern_cn.mp4', fps=15)
