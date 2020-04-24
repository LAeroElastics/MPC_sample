import time
import cvxpy as cp
import numpy as np

Xu = -0.00643
Zu = -0.0941
Mu = -0.000222
Xw = 0.0263
Zw = -0.624
Mw = -0.00153
Mq = -0.668
Mde = -2.08
U0 = 830.0
g = 32.18
c = 27.31

A = np.array([
    [Xu, Xw, -g, 0.0],
    [Zu, Zw, 0.0, U0],
    [0.0, 0.0, 0.0, 1.0],
    [Mu, Mw, 0.0, Mq]
])

B = np.array([
    [0.0],
    [0.0],
    [0.0],
    [Mde]
])

x_0 = np.array([
    [600.0],
    [100.0],
    [-0.01],
    [0.0]
]).reshape([4, ])

num_x = 4
num_u = 1
delta_t = 0.01

A = np.eye(num_x) + A * delta_t
B = B * delta_t

H_u = 100
H_p = H_u + 1
rho = 10e-4

x = cp.Variable((num_x, H_p))  # u, w, theta, q
u = cp.Variable((num_u, H_u))

c1 = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])

c2 = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])

k = 0
while k < 2000:

    cost = 0
    constr = []
    for t in range(H_u):
        cost += cp.sum_squares(x[2, t + 1] - x[1, t + 1] / U0) + cp.sum_squares(rho * u[:, t])
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t],
                   cp.norm(u[:, t], 'inf') <= 1.0]

    constr += [(x[2, H_u] - x[1, H_u] / U0) == 0, x[:, 0] == x_0.reshape([4, ])]
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(verbose=False)

    opt_u = np.array(u.value)
    x_next = A @ x_0.reshape([4, 1]) + B * opt_u[0, 0]
    x_0 = x_next
    print(x_0.reshape(-1), (x_0[2] - x_0[1] / U0), opt_u[0, 0], cost.value)  # x, x_dot, theta, theta_dot, cost
    # print(cost.value)
    k += 1
