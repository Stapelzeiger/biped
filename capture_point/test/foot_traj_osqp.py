import numpy as np
import scipy.sparse as spa
import osqp
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

Ts = 0.3
dt = 0.01
foot_height_keep_STF = 0.2
N = int(Ts/dt)
print('Nb steps = ', N)

nb_total_variables = 3 * N # px, vx, zx
print(nb_total_variables, ' variables')

# ======== Create P, q matrices ========
block = np.zeros((3, 3))
block[-1, -1] = 1
P = spa.block_diag([block] * N, format='csc')
q = np.zeros((nb_total_variables, 1))

# ======== Create A matrix ========
# boundary points
A_eq_pos_vel_desired = csr_matrix((4, nb_total_variables))
A_eq_pos_vel_desired[0, 0] = 1
A_eq_pos_vel_desired[1, 1] = 1
A_eq_pos_vel_desired[2, -3] = 1
A_eq_pos_vel_desired[3, -2] = 1

# dynamics points
block_dynamics = np.zeros((2, 6))
block_dynamics[0, 0] = 1
block_dynamics[0, 1] = dt
block_dynamics[0, 3] = -1
block_dynamics[1, 1] = 1
block_dynamics[1, 2] = dt
block_dynamics[1, 4] = -1
A_dynamics_size = (2 * N - 2, 3 * N)
A_dynamics = lil_matrix(A_dynamics_size)

j = 0
for i in range(int((2 * N - 2)/2)):
    A_dynamics[j : j + 2, i * 3 : i * 3 + 6] = block_dynamics
    j = j + 2

# limits for vel and acc
block = np.zeros((2, 3))
block[0, 1] = 1
block[1, 2] = 1
A_limits = spa.block_diag([block]*N, format='csc')

# raise foot
T_keep = 33/100*Ts
T_start_keep = 33/100*Ts
T_end_keep = T_keep + T_start_keep
n_keep = int(T_keep/dt)
n_start_keep = int(T_start_keep/dt)
n_end_keep = n_keep + n_start_keep

A_keep_foot = lil_matrix((n_keep, nb_total_variables))
j = n_start_keep
for i in range(n_keep):
    A_keep_foot[i, 3 * j] = 1
    j = j + 1

A_total = spa.vstack([A_eq_pos_vel_desired,
                    A_dynamics,
                    A_limits])
                    # A_keep_foot], format='csc')

# ======== Create l, u matrices ========
# boundary points
p0_desired = 1.0
v0_desired = 0.0
p_N_desired = 0.0
v_N_desired = 0.0
l_boundary_pts = np.zeros((4, 1))
u_boundary_pts = np.zeros((4, 1))
l_boundary_pts[0:2] = np.array([[p0_desired], [v0_desired]])
u_boundary_pts[0:2] = np.array([[p0_desired], [v0_desired]])
l_boundary_pts[2:4] = np.array([[p_N_desired], [v_N_desired]])
u_boundary_pts[2:4] = np.array([[p_N_desired], [v_N_desired]])

# dynamics points
l_dynamics = np.zeros((2 * N - 2, 1))
u_dynamics = np.zeros((2 * N - 2, 1))

# limits
v_max = 10.0
a_max = 20.0
l_limits = np.zeros((2 * N, 1))
u_limits = np.zeros((2 * N, 1))
l_limits[0::2] = -v_max
l_limits[1::2] = -a_max
u_limits[0::2] = v_max
u_limits[1::2] = a_max

# keep foot
l_keep = foot_height_keep_STF*np.ones((n_keep, 1))
u_keep = foot_height_keep_STF*np.ones((n_keep, 1))

l_total = np.vstack([l_boundary_pts,
                     l_dynamics,
                     l_limits])
                    #  l_keep])
u_total = np.vstack([u_boundary_pts,
                     u_dynamics,
                     u_limits])
                    #  u_keep])

prob = osqp.OSQP()
max_iter = 20000
eps_abs = 1.0e-03
eps_rel = 1.0e-03
prob.setup(P, q, A_total, l_total, u_total, max_iter=max_iter, eps_abs=eps_abs, eps_rel=eps_rel)

results = prob.solve()

x_opt = results.x
pos_opt = x_opt[::3]
vel_opt = x_opt[1::3]
acc_opt = x_opt[2::3]

print(pos_opt)

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 22,
    'font.family': 'serif',
    'mathtext.default': 'regular',
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 22,
    'lines.linewidth': 2,
})

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
t = np.arange(0, Ts, dt)
axs[0].plot(t, pos_opt, '.-r', label='pos')
axs[0].plot(0, p0_desired, 'o')
axs[0].plot(t[-1], p_N_desired, 'o')
axs[1].plot(t, vel_opt, '.-g', label='vel')
axs[1].plot(0, v0_desired, 'o')
axs[1].plot(t[-1], v_N_desired, 'o')
axs[2].plot(t, acc_opt, '.-b', label='acc')
[a.legend() for a in axs]
plt.show()
