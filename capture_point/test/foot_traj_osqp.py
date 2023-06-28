import numpy as np
import scipy.sparse as spa
import osqp

Ts = 0.03
dt = 0.01
N = int(Ts/dt)
print('Nb steps = ', N)

nb_total_variables = 3*N

A_step = np.array([[1, dt], [0, 1]]) # vx[n+1] = vx[n] + ax[n]*dt
B_step = np.array([[0.5*dt**2], [dt]]) # px[n+1] = px[n] + vx[n]*dt = px[n] + vx[n]*dt + 0.5*ax[n]*dt^2

A = np.kron(np.eye(N), np.eye(2))
B = np.kron(np.eye(N), B_step)


# Define initial conditions
p0 = np.zeros((2,1))  # initial position and velocity
pN = np.array([[1],[0]])  # final position and velocity



block = np.zeros((1, N))
block[0, -1] = 1

P = spa.block_diag((block, block, block))
print('P matrix = \n', P.toarray())
q = np.zeros((nb_total_variables, 1))



A_eq = spa.vstack([
    spa.hstack([A, -B]),
    spa.hstack([np.zeros((2, (N-1)*2)), np.eye(2), np.zeros((2, N))])
])

print('Aeq', A_eq)

l = np.vstack([p0, np.zeros(((N-1)*2, 1))])
u = np.vstack([np.zeros(((N-1)*2, 1)), pN])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A_eq, l, u)

# Solve problem
results = prob.solve()

# The optimal accelerations, velocities, and positions are in the x attribute of the results object
ax_opt = results.x[::2]
vx_opt = results.x[1::2]
px_opt = np.cumsum(vx_opt)*dt