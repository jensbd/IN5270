import sympy as sym
import numpy as np
V, t, I, w, dt, b, c, d, a = sym.symbols('V t I w dt b c d a')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = DtDt(u, dt) + u(t)*w**2 - f
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    R = u(dt) - (f.subs(t, 0) - u(t).subs(t, 0)*w**2)*(dt**2)/2 - u(t).subs(t, 0) - dt*sym.diff(u(t), t).subs(t, 0)
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t+dt)-2*u(t)+u(t-dt))/(dt**2)

def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print '=== Testing exact solution: %s ===' % u
    print "Initial conditions u(0)=%s, u'(0)=%s:" % \
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print 'residual step1:', residual_discrete_eq_step1(u)
    print 'residual:', residual_discrete_eq(u)
    test_quadratic()

def linear():
    main(lambda t: V*t + I)

def quadratic():
    main(lambda t: b*t**2+c*t+d)

def cubic():
    main(lambda t: a*t**3 + b*t**2+c*t+d)
#huh
def solver(func, I, w, V):
    T = 1e-11
    dt = 1e-12
    N = int(T/dt)
    t = np.linspace(0,T,N+1)
    u = np.zeros(N+1)
    u[0] = I
    u[1] = (func(0) - I*w**2)*0.5*dt**2 + I + V*dt
    for i in range(1, N):
        u[i+1] = (dt**2)*(func(t[i]) + u[i]*w**2) + 2*u[i] - u[i-1]
    return u, t

def test_quadratic():
    global I, w, V, b, d
    I = 0.5
    w = 2
    V = 1.5
    b = 3
    u_e = lambda t: b*t**2 + V*t + I
    f = ode_source_term(u_e)
    f_func = sym.lambdify(t, f)
    ulist, tlist = solver(f_func, I, w, V)
    tol = 1e-13
    error_array = np.abs(ulist - u_e(tlist))
    max_error = np.max(error_array)
    assert max_error < tol

if __name__ == '__main__':
    print("Linear\n")
    linear()
    print("--------------------------------\nQuadratic\n")
    quadratic()
    print("--------------------------------\nCubic\n")
    cubic()
