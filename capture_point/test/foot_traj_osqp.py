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

nb_total_variables_per_coordinate = 3 * N # px, vx, zx
nb_total_variables = 3 * nb_total_variables_per_coordinate # x, y, z
print(nb_total_variables, ' variables')

# ======== Create P, q matrices ========
opt_weight_desired_pos = 5000
opt_weight_desired_vel = 4000
p_N_des_x = 0.2
p_N_des_y = 0.3
p_N_des_z = 0.1
v_N_des = 0.0
block = np.zeros((3, 3))
block[-1, -1] = 1
P = np.zeros((nb_total_variables, nb_total_variables))

# put the 3x3 block on the diagonals of P
for i in range(nb_total_variables_per_coordinate):
    P[i*3:(i+1)*3, i*3:(i+1)*3] = block

P[nb_total_variables_per_coordinate - 3, nb_total_variables_per_coordinate - 3] = opt_weight_desired_pos
P[2*nb_total_variables_per_coordinate - 3, 2*nb_total_variables_per_coordinate - 3] = opt_weight_desired_pos
P[3*nb_total_variables_per_coordinate - 3, 3*nb_total_variables_per_coordinate - 3] = opt_weight_desired_pos

P[nb_total_variables_per_coordinate - 2, nb_total_variables_per_coordinate - 2] = opt_weight_desired_vel
P[2*nb_total_variables_per_coordinate - 2, 2*nb_total_variables_per_coordinate - 2] = opt_weight_desired_vel
P[3*nb_total_variables_per_coordinate - 2, 3*nb_total_variables_per_coordinate - 2] = opt_weight_desired_vel

P = csr_matrix(P)

q = np.zeros((nb_total_variables, 1))
q[nb_total_variables_per_coordinate - 3] = -2 * opt_weight_desired_pos * p_N_des_x
q[2*nb_total_variables_per_coordinate - 3] = -2 * opt_weight_desired_pos * p_N_des_y
q[3*nb_total_variables_per_coordinate - 3] = -2 * opt_weight_desired_pos * p_N_des_z

q[nb_total_variables_per_coordinate - 2] = -2*opt_weight_desired_vel*v_N_des
q[2*nb_total_variables_per_coordinate - 2] = -2*opt_weight_desired_vel*v_N_des
q[3*nb_total_variables_per_coordinate - 2] = -2*opt_weight_desired_vel*v_N_des

# q = csr_matrix(q)

# ======== Create A matrix ========
# boundary points
A_eq_pos_vel_des = csr_matrix((6, nb_total_variables))
A_eq_pos_vel_des[0, 0] = 1
A_eq_pos_vel_des[1, 1] = 1
A_eq_pos_vel_des[2, nb_total_variables_per_coordinate] = 1
A_eq_pos_vel_des[3, nb_total_variables_per_coordinate + 1] = 1
A_eq_pos_vel_des[4, 2*nb_total_variables_per_coordinate] = 1
A_eq_pos_vel_des[5, 2*nb_total_variables_per_coordinate + 1] = 1

# dynamics points
block_dynamics = np.zeros((2, 6))
block_dynamics[0, 0] = 1
block_dynamics[0, 1] = dt
block_dynamics[0, 3] = -1
block_dynamics[1, 1] = 1
block_dynamics[1, 2] = dt
block_dynamics[1, 4] = -1
A_dynamics_size = (2 * N - 2, 3 * N)
A_dynamics_per_coordinate = np.zeros(A_dynamics_size)

j = 0
for i in range(int((2 * N - 2)/2)):
    A_dynamics_per_coordinate[j : j + 2, i * 3 : i * 3 + 6] = block_dynamics
    j = j + 2

size_rows_A = A_dynamics_per_coordinate.shape[0]
size_cols_A = A_dynamics_per_coordinate.shape[1]
print(size_cols_A, ' columns')
print(size_rows_A, ' rows')
A_dynamics = np.zeros((3 * size_rows_A, nb_total_variables))

for i in range(3):
    A_dynamics[i * size_rows_A : i * size_rows_A + size_rows_A, i * size_cols_A : i * size_cols_A + size_cols_A] = A_dynamics_per_coordinate
A_dynamics = csr_matrix(A_dynamics)

# limits for vel and acc
block = np.zeros((2, 3))
block[0, 1] = 1
block[1, 2] = 1
A_limits = np.zeros((nb_total_variables_per_coordinate * 2, nb_total_variables))

for i in range(nb_total_variables_per_coordinate):
    A_limits[2*i:2*(i+1), 3*i:3*(i+1)] = block

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

A_total = spa.vstack([A_eq_pos_vel_des,
                    A_dynamics,
                    A_limits])
                    # A_keep_foot], format='csc')

# ======== Create l, u matrices ========
# boundary points
p0_desired = 0.0
v0_desired = 0.0
l_boundary_pts_per_coordinate = np.zeros((2, 1))
u_boundary_pts_per_coordinate = np.zeros((2, 1))
l_boundary_pts_per_coordinate[0:2] = np.array([[p0_desired], [v0_desired]])
u_boundary_pts_per_coordinate[0:2] = np.array([[p0_desired], [v0_desired]])

l_boundary_pts = np.vstack([l_boundary_pts_per_coordinate] * 3)
u_boundary_pts = np.vstack([u_boundary_pts_per_coordinate] * 3)

# dynamics points
l_dynamics_per_coordinate = np.zeros((2 * N - 2, 1))
u_dynamics_per_coordinate = np.zeros((2 * N - 2, 1))
l_dynamics = np.vstack([l_dynamics_per_coordinate] * 3)
u_dynamics = np.vstack([u_dynamics_per_coordinate] * 3)

# limits
v_max = 100.0
a_max = 300.0
l_limits_per_coordinate = np.zeros((2 * N, 1))
u_limits_per_coordinate = np.zeros((2 * N, 1))
l_limits_per_coordinate[0::2] = -v_max
l_limits_per_coordinate[1::2] = -a_max
u_limits_per_coordinate[0::2] = v_max
u_limits_per_coordinate[1::2] = a_max

l_limits = np.vstack([l_limits_per_coordinate] * 3)
u_limits = np.vstack([u_limits_per_coordinate] * 3)

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
eps_abs = 1.0e-05
eps_rel = 1.0e-05
prob.setup(P, q, A_total, l_total, u_total, max_iter=max_iter, eps_abs=eps_abs, eps_rel=eps_rel)

results = prob.solve()

x_opt = results.x
print(x_opt)
pos_opt_x = x_opt[::3][0:N]
pos_opt_y = x_opt[::3][N : 2 * N]
pos_opt_z = x_opt[::3][2 * N: 3 * N]

vel_opt_x = x_opt[1::3][0:N]
vel_opt_y = x_opt[1::3][N:2*N]
vel_opt_z = x_opt[1::3][2*N:3*N]

acc_opt_x = x_opt[2::3][0:N]
acc_opt_y = x_opt[2::3][N:2*N]
acc_opt_z = x_opt[2::3][2*N:3*N]


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

fig, axs = plt.subplots(3, 3, figsize=(30, 10))
t = np.arange(0, Ts, dt)
axs[0, 0].plot(t, pos_opt_x, '.-r', label='pos')
axs[0, 0].plot(0, p0_desired, 'o')
axs[0, 0].plot(t[-1], p_N_des_x, 'o')
axs[1, 0].plot(t, vel_opt_x, '.-g', label='vel')
axs[1, 0].plot(0, v0_desired, 'o')
# axs[1].plot(t[-1], v_N_desired, 'o')
axs[2, 0].plot(t, acc_opt_x, '.-b', label='acc')

axs[0, 1].plot(t, pos_opt_y, '.-r', label='pos')
axs[0, 1].plot(0, p0_desired, 'o')
axs[0, 1].plot(t[-1], p_N_des_y, 'o')
axs[1, 1].plot(t, vel_opt_y, '.-g', label='vel')
axs[1, 1].plot(0, v0_desired, 'o')
# axs[1].plot(t[-1], v_N_desired, 'o')
axs[2, 1].plot(t, acc_opt_y, '.-b', label='acc')

axs[0, 2].plot(t, pos_opt_z, '.-r', label='pos')
axs[0, 2].plot(0, p0_desired, 'o')
axs[0, 2].plot(t[-1], p_N_des_z, 'o')
axs[1, 2].plot(t, vel_opt_z, '.-g', label='vel')
axs[1, 2].plot(0, v0_desired, 'o')
# axs[1].plot(t[-1], v_N_desired, 'o')
axs[2, 2].plot(t, acc_opt_z, '.-b', label='acc')

plt.tight_layout()
# [a.legend() for a in axs]
plt.show()
