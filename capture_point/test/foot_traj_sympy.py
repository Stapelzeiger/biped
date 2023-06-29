from sympy import *
from sympy.matrices import BlockMatrix, Matrix, ones, zeros
import sympy as sp

Ts = 0.1
dt = 0.01
N = int(Ts/dt)
nb_total_variables = 3 * N # px, vx, zx

init_printing()
variables = []
for i in range(N):
    p, v, a = symbols(f'p{i} v{i} a{i}')
    variables.extend([p, v, a])
pva_vector = Matrix(variables)
print('------------------- Optimization Variable -------------------')
pprint(pva_vector.T)

print('------------------- P Mat -------------------')
block = zeros(3, 3)
block[-1, -1] = 1
P_blocks = [[block if i==j else zeros(3,3) for i in range(N)] for j in range(N)]
P = BlockMatrix(P_blocks).as_explicit()
vPv = pva_vector.T * P * pva_vector
print('P*v = ', vPv)


A_eq_pos_vel_desired = sp.zeros(4, nb_total_variables)
A_eq_pos_vel_desired[0, 0] = 1
A_eq_pos_vel_desired[1, 1] = 1
A_eq_pos_vel_desired[2, -3] = 1
A_eq_pos_vel_desired[3, -2] = 1

result_vector = A_eq_pos_vel_desired*pva_vector
print('------------------- Boundary Points Equality Contraints -------------------')
print('A_eq_pos_vel_desired * v = ')
pprint(result_vector)

dt_symbol = symbols('dt')
block_dynamics = zeros(2, 6)
block_dynamics[0, 0] = 1
block_dynamics[0, 1] = dt_symbol
block_dynamics[0, 3] = -1
block_dynamics[1, 1] = 1
block_dynamics[1, 2] = dt_symbol
block_dynamics[1, 4] = -1
A_dynamics = zeros(2 * N - 2, 3 * N)

j = 0
for i in range(int((2 * N - 2)/2)):
    A_dynamics[j : j + 2, i * 3: i * 3 + 6] = block_dynamics
    j = j + 2
mult = A_dynamics * pva_vector
print('------------------- Dynamic Contraints -------------------')
pprint(mult)


block = zeros(2, 3)
block[0, 1] = 1
block[1, 2] = 1
A_limits = zeros(N * 2, N * 3)

for i in range(N):
    A_limits[2*i:2*(i+1), 3*i:3*(i+1)] = block


print('------------------- Limits -------------------')
pprint((A_limits * pva_vector).T)

T_keep = 30/100 * Ts
T_start_keep = 30/100 * Ts
T_end_keep = T_keep + T_start_keep
n_keep = int(T_keep / dt)
n_start_keep = int(T_start_keep / dt)
n_end_keep = int(n_keep + n_start_keep)

print(n_keep)
print(n_start_keep)
print(n_end_keep)

# A_keep_foot = SparseMatrix(n_keep, nb_total_variables)
A_keep_foot = SparseMatrix.zeros(n_keep, nb_total_variables)
j = n_start_keep
for i in range(n_keep):
    A_keep_foot[i, 3 * j] = 1
    j = j + 1
print('------------------- Keep Foot -------------------')
pprint(A_keep_foot)
pprint((A_keep_foot * pva_vector).T)
