import numpy as np
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
dt = 0.25
end_t = 200
time_steps = int(end_t / dt)
print(time_steps)
c = dt / (h * h)

# Initial condition
U_init = np.zeros((Ny, Nx))
V_init = np.zeros((Ny, Nx))
U_init[:, :] = 1
V_init[51:60, 51:70] = 1
V_init[61:80, 71:80] = 1

U = np.zeros((time_steps, Ny, Nx))
V = np.zeros((time_steps, Ny, Nx))
U[0] = np.copy(U_init)
V[0] = np.copy(V_init)

# Generate pattern using own laplace function
for i in range(1, time_steps):
    U[i] = Du * c * my_laplacian(U[i - 1])
    V[i] = Dv * c * my_laplacian(V[i - 1])

    U[i] += -dt * np.multiply(U[i - 1], np.square(V[i - 1])) \
        + dt * ff * (1 - U[i - 1]) + U[i - 1]
    V[i] += dt * np.multiply(U[i - 1], np.square(V[i - 1])) \
        - dt * (kk + ff) * V[i - 1] + V[i - 1]

np.save('../ndarr_UV/U_fwd', U)
np.save('../ndarr_UV/V_fwd', V)

U_fwd = np.load('../ndarr_UV/U_fwd.npy')
V_fwd = np.load('../ndarr_UV/V_fwd.npy')

plot_pattern(U_fwd, V_fwd, time_steps - 1, time_steps - 1, h, dt, filled=True)

# ani = animate_pattern(U_fwd, V_fwd, h, dt, Nsteps=time_steps, Nout=10)
# ani.save('../ani/pattern_fwd.mp4', fps=15)
