import numpy as np
import scipy.sparse as spa
import osqp

Ts = 0.04
dt = 0.01
N = int(Ts/dt)
print('Nb steps = ', N)

nb_total_variables = 3*N
print(nb_total_variables, ' variables')

# ======== Create P, q matrices ========
block = np.zeros((3, 3))
block[-1, -1] = 1
P = spa.block_diag([block]*N)
print('P matrix = \n', P.toarray())
q = np.zeros((nb_total_variables, 1))

# ======== Create A matrix ========
block = np.zeros((2, 3))
block[0, 0] = 1
block[1, 1] = 1
A_eq_pos_vel_desired = spa.block_diag([block]*N)

block_dynamics = np.zeros((2, 5))
block_dynamics[0, 0] = 1
block_dynamics[0, 1] = dt
block_dynamics[0, 2] = -1
block_dynamics[1, 1] = 1
block_dynamics[1, 3] = dt
block_dynamics[1, 4] = -1

A_dynamics = spa.block_diag([block_dynamics]*N)
print(A_dynamics.toarray())
print(A_dynamics.shape)

# A_total = spa.vstack([A_eq_pos_vel_desired, A_dynamics])

A_total = A_eq_pos_vel_desired.copy()
print('A matrix = \n', A_total.toarray())

# ======== Create l, u matrices ========

p0_desired = 0.0
v0_desired = 0.0
p_N_desired = 1.0
v_N_desired = 0.0

l = np.zeros((2*N, 1))
u = np.zeros((2*N, 1))

l[0:2] = np.array([[p0_desired], [v0_desired]])
u[0:2] = np.array([[p0_desired], [v0_desired]])
l[2 * N - 2 :2 * N] = np.array([[p_N_desired], [v_N_desired]])
u[2 * N - 2 :2 * N] = np.array([[p_N_desired], [v_N_desired]])

l_total = l.copy()
u_total = u.copy()

prob = osqp.OSQP()
prob.setup(P, q, A_total, l_total, u_total)
results = prob.solve()

x_opt = results.x
pos_opt = x_opt[::3]
vel_opt = x_opt[1::3]
acc_opt = x_opt[2::3]

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
axs[0].plot(t, pos_opt, 'r', label='pos')
axs[0].plot(0, p0_desired, 'o')
axs[0].plot(t[-1], p_N_desired, 'o')
axs[1].plot(t, vel_opt, 'g', label='vel')
axs[1].plot(0, v0_desired, 'o')
axs[1].plot(t[-1], v_N_desired, 'o')
axs[2].plot(t, acc_opt, 'b', label='acc')
[a.legend() for a in axs]
plt.show()
