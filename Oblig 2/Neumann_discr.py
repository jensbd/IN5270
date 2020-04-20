
import numpy as np
import sympy as sym


x, t = sym.symbols("x t")
global L
L = 1
global w
w = np.pi


def u_exact(L, omega=np.pi):
    return lambda x, t: sym.cos(sym.pi * x / L) * sym.cos(omega * t)


def find_source_term(u, q):
    return sym.diff(u(x, t), t, 2) - sym.diff(q(x) * sym.diff(u(x, t), x), x)


def solver(q_lambda, f, T, Nx, dt, task):
    Nt = int(T / dt)
    It = range(0, Nt + 1)
    dt = T / Nt

    # Set dx based on the stability criterion
    x_tmp = np.linspace(0, L, Nx + 1)
    beta = 0.9
    q_max = np.max(get_q_values(L, q_lambda))

    dx = dt * q_max / beta

    Nx = int(round(L / dx))
    Ix = range(0, Nx + 1)

    C2 = (dt / dx) ** 2

    u_exact = np.zeros(Nx + 1)
    u = np.zeros(Nx + 1)
    u_1 = np.zeros(Nx + 1)
    u_2 = np.zeros(Nx + 1)

    x_array = np.linspace(0, L, Nx + 1)
    t_array = np.linspace(0, T, Nt + 1)


    error = np.zeros(Nt + 1)

    # Initial condition
    u_1 = np.cos(np.pi * x_array / L)
    #getting all q-values in array
    q = get_q_values(L,q_lambda,Nx)

    #special rule for inner mesh grid values at first step
    for i in Ix[1:-1]:
        u[i] = u_1[i] + 0.25 * C2 * ((q[i] + q[i + 1]) * (u_1[i + 1] - u_1[i])
               - (q[i] + q[i - 1]) * (u_1[i] - u_1[i - 1])) + 0.5 * dt ** 2 \
               * f(x_array[i], t_array[0])

        u_exact[i] = np.cos(np.pi * x_array[i] / L) * np.cos(w * t_array[0])

    #boundary values
    if task == 'a':
        i = 0
        u[i] = u_1[i] + q[i] * C2 * (u_1[i + 1] - u_1[i]) + 0.5 * dt ** 2 * f(x_array[i], t_array[0])

        i = Nx
        u[i] = u_1[i] + q[i] * C2 * (u_1[i - 1] - u_1[i]) + 0.5 * dt ** 2 * f(x_array[i], t_array[0])
    if task == 'b':
        i = 0
        ip1 = i + 1
        u[i] = u_1[i] + C2 * 0.5 * (q[ip1] + q[i]) * (u_1[ip1] - u_1[i]) + 0.5 * dt ** 2 * f(x_array[i], t_array[0])

        i = Nx
        im1 = i - 1
        u[i] = u_1[i] + C2 * 0.5 * (q[im1] + q[i]) * (u_1[im1] - u_1[i]) + 0.5 * dt ** 2 * f(x_array[i], t_array[0])
    if task == 'c':
        i = 0
        u[i] = u[i+1]
        i = Nx
        u[i] = u[i-1]
    if task == 'd':
        i = 0
        ip1 = i + 1
        u[i] = u_1[i] + C2 * 0.25 * (q[ip1] + q[i]) * (u_1[ip1] - u_1[i]) + 0.5 * dt ** 2 * f(x_array[i], t_array[0])
        i = Nx
        im1 = i - 1
        u[i] = u_1[i] + C2 * 0.25 * (q[im1] + q[i]) * (u_1[im1] - u_1[i]) + 0.5 * dt ** 2 * f(x_array[i], t_array[0])


    error[0] = np.sum((u_exact - u) ** 2)
    u_2, u_1, u = u_1, u, u_2


    # Time loop
    for n in It[1:-1]:
        # Compute internal points
        u_exact = np.cos(np.pi * x_array / L) * np.cos(w * t_array[n])

        u[1:-1] = 2 * u_1[1:-1] - u_2[1:-1] + C2 * (
                0.5 * ( q[1:-1] + q[2:]) * (u_1[2:] - u_1[1:-1]) - 0.5 * (q[1:-1]
                + q[:-2]) * (u_1[1:-1] - u_1[:-2])) + dt ** 2\
                * f(x_array[1:-1], t_array[n])


        if task == 'a':
            i = 0
            ip1 = i + 1
            u[i] = 2 * u_1[i] - u_2[i] + 2 * q[i] * C2 * (u_1[ip1] - u_1[i]) + dt ** 2 * f(x_array[i], t_array[n])
            i = Nx
            im1 = i - 1
            u[i] = 2 * u_1[i] - u_2[i] + 2 * q[i] * C2 * (u_1[im1] - u_1[i]) + dt ** 2 * f(x_array[i], t_array[n])

        if task == 'b':
            i = 0
            ip1 = i + 1
            u[i] = 2 * u_1[i] - u_2[i] + C2 * (q[ip1] + q[i]) * (u_1[ip1] - u_1[i]) + dt ** 2 * f(x_array[i], t_array[n])
            i = Nx
            im1 = i - 1
            u[i] = 2 * u_1[i] - u_2[i] + C2 * (q[im1] + q[i]) * (u_1[im1] - u_1[i]) + dt ** 2 * f(x_array[i], t_array[n])
        if task == 'c':
            i = 0
            u[i] = u[i+1]
            i = Nx
            u[i] = u[i-1]
        if task == 'd':
            i = 0
            ip1 = i + 1
            u[i] = 2 * u_1[i] - u_2[i] + C2 * 0.5 * (q[ip1] + q[i]) * (u_1[ip1] - u_1[i]) + dt ** 2 * f(x_array[i], t_array[n])
            i = Nx
            im1 = i - 1
            u[i] = 2 * u_1[i] - u_2[i] + C2 * 0.5 * (q[im1] + q[i]) * (u_1[im1] - u_1[i]) + dt ** 2 * f(x_array[i], t_array[n])

        error[n] = np.sqrt(dx * np.sum((u_exact - u) ** 2))
        u_2, u_1, u = u_1, u, u_2

    u = u_1
    return error[n]

def get_q_values(L, q, Nx=None):
    if Nx:
        x_tmp = np.linspace(0, L, Nx + 1)
    else:
        x_tmp = np.linspace(0, L)
    q_tmp = np.zeros(len(x_tmp))
    for i in range(len(x_tmp)):
        q_tmp[i] = q(x_tmp[i])
    return q_tmp


def main():
    u_e = u_exact(L)

    q_a = (lambda x: 1 + (x - L / 2) ** 4)
    source_term_a = find_source_term(u_e, q_a)
    f_a = sym.lambdify((x,t), source_term_a)

    h_values = [0.05, 0.025]
    errors = []

    for h in h_values:
        dt = h
        error = solver(q_a, f_a, T=1, Nx=100, dt=dt, task = 'a')
        errors.append(error)

    r = np.log(errors[1] / errors[0]) / np.log(h_values[1] / h_values[0])
    print("convergence rate task:  a :", r)


    # Setup up initial condition, q(x) and source terms f(x,t)
    q = lambda x: 1 + sym.cos(sym.pi * x / L)

    source_term = find_source_term(u_e, q)
    f_func = sym.lambdify((x, t), source_term)

    tasks = ['b', 'c', 'd']

    for task in tasks:
        errors = []

        for h in h_values:
            dt = h
            error = solver(q, f_func, T=1, Nx=100, dt=dt, task = task)

            errors.append(error)

        r = np.log(errors[1] / errors[0]) / np.log(h_values[1] / h_values[0])
        print("convergence rate task: ",task,":", r)

if __name__ == '__main__':
    main()
    """
    Running:
    convergence rate task:  a : 2.04997544557
    convergence rate task:  b : 2.04311693124
    convergence rate task:  c : 0.923281707362
    convergence rate task:  d : 0.994650398931
    """
