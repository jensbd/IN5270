"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0
  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from __future__ import print_function
from fenics import *
import numpy as np
from tqdm import tqdm
#Suppress general progress messages
set_log_level(30)
def picard(N, intervals, alpha, rho, I, f = False, plotting = False):

    def boundary(x, on_boundary):
        return False

    # Different meshes for 1D, 2D, 3D
    mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
    d = len(intervals)
    mesh = mesh_classes[d-1](*intervals)

    V = FunctionSpace(mesh, 'P', 1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)


    T = 1.0
    dt = (T/N)*(T/N)
    t = 0
    u_1 = project(I,V)
    u_k = u_1


    if f:
        f = Expression('-rho*pow(x[0],3)/3. + rho*pow(x[0],2)/2. + 8*pow(t,3)*pow(x[0],7)/9.- 28*pow(t,3)*pow(x[0],6)/9. + 7*pow(t,3)*pow(x[0],5)/2. - 5*pow(t,3)*pow(x[0],4)/4. + 2*t*x[0] - t',rho=rho,t=t, degree = 1)
        f.t=0
    else:
        f = Constant('0')

    F = u*v*dx + dot(dt/rho*alpha(u_k)*grad(u),grad(v))*dx - u_1*v*dx - dt/rho*f*v*dx
    a = lhs(F)
    L = rhs(F)


    while t <= T:
        t += dt
        f.t = t
        solve(a==L, u_)
        if plotting:
            plot(u_,rescale=False,interactive=True) # Press q to proceed to next timestep
        u_1.assign(u_)  # Updating solution
        u_k.assign(u_)            # Updating solution for the picard iteration which is u_1
    return u_, t, V, dt




def task_d():
    print("-------\nTask d: \n-------")
    def alpha(u):
        return 1
    rho = 1.0
    N  = 8
    intervals = [8,8]
    I = Expression("1",degree = 1)
    u_, t, V,dt = picard(N, intervals,alpha, rho,I)

    u_exact = Expression('1',degree = 1)
    u_e = project(u_exact,V)
    diff = max(abs(u_e.vector().get_local() - u_.vector().get_local()))
    tol = 1.0E-5
    print("Max absolute difference between numerical and exact solution:",diff)
    if diff <tol:
        print ("Passed, the solutions are satisfactorily close")
    else:
        print("Failed, the solutions are not close enough")

def task_e():
    print("-------\nTask e: \n-------")
    rho = 1.0
    I = Expression("cos(pi*x[0])",degree = 1)
    def alpha(u):
        return 1
    k = []
    h = []
    for i in tqdm([10,15,20,30,50]):
        u_, t, V,dt = picard(i,[i,i], alpha, rho,I)
        u_exact = Expression('exp(-(pi*pi*t))*cos(pi*x[0])',t=t, degree = 1)
        h.append(dt)
        u_e = project(u_exact,V)
        e = (u_e.vector().get_local() - u_.vector().get_local())
        E = np.sqrt(np.sum(e**2)/u_.vector().get_local().size)
        k.append(E/dt)

    for i in range(len(k)):
        print("h = %3.5f, E/h = %3.7e" %( h[i], k[i] ))

def task_f():
    print("-------\nTask f: \n-------")
    rho = 1.0
    I = Constant('0')

    def alpha(u):
        return (1+u*u)

    k = []
    h = []
    for i in tqdm([10,20,30,50,100]):
        u_, t, V,dt = picard(i,[i], alpha, rho,I,f= True)
        u_exact = Expression('t*pow(x[0],2)*(0.5-x[0]/3.)',t=t, degree = 1)
        h.append(dt)
        u_e = project(u_exact,V)
        e = (u_e.vector().get_local() - u_.vector().get_local())
        E = np.sqrt(np.sum(e**2)/u_.vector().get_local().size)
        k.append(E/dt)
    for i in range(len(k)):
        print ("h = %3.5f, E/h = %3.7e" %( h[i], k[i] ))

task_d()
task_e()
task_f()
