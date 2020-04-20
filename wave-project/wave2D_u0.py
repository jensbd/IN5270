"""
2D wave equation solved by finite differences::
  dt, cpu_time = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T,
                        user_action=None, version='scalar',
                        stability_safety_factor=1)
Solve the 2D wave equation u_tt = u_xx + u_yy + f(x,t) on (0,L) with
u=0 on the boundary and initial condition du/dt=0.
Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).
dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.
I, V, f are functions: I(x,y), V(x,y), f(x,y,t). V and f
can be specified as None or 0, resulting in V=0 and f=0.
user_action: function of (u, x, y, t, n) called at each time
level (x and y are one-dimensional coordinate vectors).
This function allows the calling code to plot the solution,
compute errors, etc.
"""
import glob, os
import time, sys
import sympy as sym
#from scitools.std import *
from numpy import *


def advance_scalar(u, u_1, u_2, f, x, y, t, b, n, dt2,dx,dy,q,
                   V=None, step1=False):
    Ix = range(1, u.shape[0]-1);  Iy = range(1, u.shape[1]-1)
    dt = sqrt(dt2)  # save
    for i in Ix:
        for j in Iy:
            if step1:
                u[i,j] = -dt*V(x[i-Ix[0]], y[j-Iy[0]])*(b*dt/2 -1 ) + u_1[i,j] \
                + ((dt2)/(4*dx**2))*((q[i,j]+q[i+1,j])*(u_1[i+1,j]-u_1[i,j]) - (q[i-1,j]+q[i,j])*(u_1[i,j]-u_1[i-1,j]) ) \
                +  (dt2 /(4*dy**2))*((q[i,j]+q[i,j+1])*(u_1[i,j+1]-u_1[i,j]) - (q[i,j-1]+q[i,j])*(u_1[i,j]-u_1[i,j-1]) ) \
                + 0.5*(dt**2) *dt2* f(x[i-Ix[0]],y[j-Iy[0]],t[n])
            else:

                u[i,j] = (b*dt/2 + 1)**(-1) * (u_2[i,j]*((b*dt/2) - 1) + 2*u_1[i,j] \
                + (dt2/(dx**2))*(0.5*(q[i,j]+q[i+1,j])*(u_1[i+1,j]-u_1[i,j]) - 0.5*(q[i-1,j]+q[i,j])*(u_1[i,j]-u_1[i-1,j]) ) \
                + (dt2/(dy**2))*(0.5*(q[i,j]+q[i,j+1])*(u_1[i,j+1]-u_1[i,j]) - 0.5*(q[i,j-1]+q[i,j])*(u_1[i,j]-u_1[i,j-1]) ) \
                + (dt**2)*f(x[i-Ix[0]],y[j-Iy[0]],t[n]) )

    j = Iy[0]
    for i in Ix:
        u[i-1,j-1] = u[i-1,j+1]
        q[i-1,j-1] = q[i-1,j+1]

    i = Ix[0]
    for j in Iy:
        u[i-1,j-1] = u[i+1,i-1]
        q[i-1,j-1] = q[i+1,j-1]

    j = Iy[-1]
    for i in Ix:
        u[i-1,j+1] = u[i-1,j-1]
        q[i-1,j+1] = q[i-1,j-1]

    i = Ix[-1]
    for j in Iy:
        u[i+1,j-1] = u[i-1,j-1]
        q[i+1,j-1] = q[i-1,j-1]

    return u



def advance_vectorized(u, u_1, u_2, f_a, x,y,t,b,dt2,dx,dy,q,
                       V=None, step1=False):

    dt = sqrt(dt2)  # save
    if step1:
        u[1:-1,1:-1] = - dt*V*(b*dt/2. -1. ) + u_1[1:-1,1:-1] \
                       + (dt2/(4*dx**2))*((q[1:-1,1:-1]+q[2:,1:-1])*(u_1[2:,1:-1]-u_1[1:-1,1:-1]) - (q[:-2,1:-1]+q[1:-1,1:-1])*(u_1[1:-1,1:-1]-u_1[:-2,1:-1]) ) \
                       + (dt2/(4*dy**2))*((q[1:-1,1:-1]+q[1:-1,2:])*(u_1[1:-1,2:]-u_1[1:-1,1:-1]) - (q[1:-1,:-2]+q[1:-1,1:-1])*(u_1[1:-1,1:-1]-u_1[1:-1,:-2]) ) \
                       + 0.5*(dt2) * f_a

    else:
        u[1:-1,1:-1] =  (b*dt/2 + 1)**(-1) * (u_2[1:-1,1:-1]*((b*dt/2) - 1) + 2*u_1[1:-1,1:-1] \
                      + (dt2/(dx**2))*(0.5*(q[1:-1,1:-1]+q[2:,1:-1])*(u_1[2:,1:-1]-u_1[1:-1,1:-1]) - 0.5*(q[:-2,1:-1]+q[1:-1,1:-1])*(u_1[1:-1,1:-1]-u_1[:-2,1:-1]) ) \
                      + (dt2/(dy**2))*(0.5*(q[1:-1,1:-1]+q[1:-1,2:])*(u_1[1:-1,2:]-u_1[1:-1,1:-1]) - 0.5*(q[1:-1,:-2]+q[1:-1,1:-1])*(u_1[1:-1,1:-1]-u_1[1:-1,:-2]) ) \
                      + (dt**2)*f_a)

    # Boundary conditions
    j = 1
    u[:,j-1] = u[:,j+1]
    q[:,j-1] = q[:,j+1]

    j = u.shape[1]-2
    u[:,j+1] = u[:,j-1]
    q[:,j+1] = q[:,j-1]

    i = 1
    u[i-1,:] = u[i+1,:]
    q[i-1,:] = q[i+1,:]

    i = u.shape[0]-2
    u[i+1,:] = u[i-1,:]
    q[i+1:,:] = q[i-1,:]

    return u


def solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T,b,q_func,
           user_action=None, version='scalar'):

    if version == 'vectorized':
        advance = advance_vectorized
    elif version == 'scalar':
        advance = advance_scalar
    else:
        raise ValueError

    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xv = x[:,newaxis]          # for vectorized function evaluations
    yv = y[newaxis,:]
    stability_limit = (1/float(c))*(1/sqrt(1/dx**2))
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        print ('error: dt=%f exceeds the stability limit %f'% (dt, stability_limit))
    Nt = int(round(T/float(dt)))
    t = linspace(0, Nt*dt, Nt+1)    # mesh points in time
    dt2 = dt**2


    # Allow f and V to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: 0) if version == 'scalar' else \
            lambda x, y, t: zeros((x.shape[0], y.shape[1]))
        # or simpler: x*y*0
    if V is None or V == 0:
        V = (lambda x, y: 0) if version == 'scalar' else \
            lambda x, y: zeros((x.shape[0], y.shape[1]))


    order = 'Fortran' if version == 'f77' else 'C'
    u   = zeros((Nx+3,Ny+3), order=order)   # solution array
    u_1 = zeros((Nx+3,Ny+3), order=order)   # solution at t-dt
    u_2 = zeros((Nx+3,Ny+3), order=order)   # solution at t-2*dt
    f_a = zeros((Nx+1,Ny+1), order=order)   # for compiled loops

    Ix = range(1, u.shape[0]-1)
    Iy = range(1, u.shape[1]-1)
    It = range(0, t.shape[0])

    q = zeros(u.shape)
    i_ = 1


    for i in x:
        j_ = 1
        for j in y:
            q[i_,j_] = q_func(i,j)
            j_ += 1
        i_ +=1

    import time; t0 = time.perf_counter()          # for measuring CPU time

    # Load initial condition into u_1

    for i in Ix:
        for j in Iy:
            u_1[i,j] = I(x[i-Ix[0]], y[j-Iy[0]])


    if user_action is not None:
        user_action(u_1[ Ix[0]:Ix[-1]+1, Iy[0]:Iy[-1]+1 ], x, xv, y, yv, t, 0)

    # Special formula for first time step
    n = 0
    # First step requires a special formula, use either the scalar
    # or vectorized version (the impact of more efficient loops than
    # in advance_vectorized is small as this is only one step)
    if version == 'scalar':
        u = advance_scalar(
            u, u_1, u_2, f, x, y, t,b,n,
            dt2,dx,dy,q,V, step1=True)

    else:
        f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
        V_a = V(xv, yv)
        u = advance_vectorized(
            u, u_1, u_2, f_a,x,y,t,b,
            dt2, dx,dy,q,V=V_a, step1=True)

    if user_action is not None:
        user_action(u[ Ix[0]:Ix[-1]+1, Iy[0]:Iy[-1] + 1 ], x, xv, y, yv, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        if version == 'scalar':
            # use f(x,y,t) function
            u = advance(u, u_1, u_2, f, x, y, t,b, n, dt2,dx, dy,q)
        else:
            f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
            u = advance(u, u_1, u_2, f_a,x,y,t,b, dt2, dx, dy,q)

        if user_action is not None:
            if user_action(u[Ix[0]:Ix[-1]+1, Iy[0]:Iy[-1]+1], x, xv, y, yv, t, n+1):
                break
        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2
    # Important to set u = u_1 if u is to be returned!
    t1 = time.perf_counter()
    # dt might be computed in this function so return the value
    return dt, t1 - t0

def quadratic(Nx, Ny, version):
    """Exact discrete solution of the scheme."""
    t = sym.symbols("t")

    def exact_solution(x, y, t):
        return 20

    def I(x, y):
        return exact_solution(x, y, 0)

    def V(x, y):
        return sym.diff(exact_solution(x, y, 0),t)

    def f(x, y, t):
        return 0

    Lx = 5;  Ly = 2
    c = 1
    dt = -1 # use longest possible steps
    T = 18
    b = 0

    q = (lambda x,y: 0)

    def assert_no_error(u, x, xv, y, yv, t, n):
        u_e = exact_solution(xv, yv, t[n])
        diff = abs(u - u_e).max()
        tol = 1e-12
        msg = 'diff=%g, step %d, time=%g' % (diff, n, t[n])
        assert diff < tol, msg

    new_dt, cpu = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T,b,q,
        user_action=assert_no_error, version=version)
    return new_dt, cpu


def test_quadratic():
    # Test a series of meshes where Nx > Ny and Nx < Ny
    versions = 'scalar', 'vectorized'
    for Nx in range(2, 6, 2):
        for Ny in range(2, 6, 2):
            for version in versions:
                print ('testing', version, 'for %d x %d mesh' %  (Nx, Ny))
                quadratic(Nx, Ny, version)

def run_efficiency(nrefinements=4):
    def I(x, y):
        return sin(pi*x/Lx)*sin(pi*y/Ly)

    Lx = 10;  Ly = 10
    c = 1.5
    T = 100
    versions = ['scalar', 'vectorized']
    print (' '*15, ''.join(['%-13s' % v for v in versions]))
    for Nx in 15, 30, 60, 120:
        cpu = {}
        for version in versions:
            dt, cpu_ = solver(I, None, None, c, Lx, Ly, Nx, Nx,
                              -1, T, user_action=None,
                              version=version)
            cpu[version] = cpu_
        cpu_min = min(list(cpu.values()))
        if cpu_min < 1E-6:
            print ('Ignored %dx%d grid (too small execution time)', (Nx, Nx))
        else:
            cpu = {version: cpu[version]/cpu_min for version in cpu}
            print ('%-15s' % '%dx%d' % (Nx, Nx))
            print (''.join(['%13.1f' % cpu[version] for version in versions]))

def gaussian(plot_method=3, version='vectorized', save_plot=True):
    """
    Initial Gaussian bell in the middle of the domain.
    plot_method=1 applies mesh function, =2 means surf, =0 means no plot.
    """
    # Clean up plot files
    #for name in glob('tmp_*.png'):
    #    os.remove(name)

    Lx = 10
    Ly = 10
    c = 1.0

    def I(x, y):
        """Gaussian peak at (Lx/2, Ly/2)."""
        return exp(-0.5*(x-Lx/2.0)**2 - 0.5*(y-Ly/2.0)**2)

    if plot_method == 3:
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        from matplotlib import cm
        plt.ion()
        fig = plt.figure()
        u_surf = None

    def plot_u(u, x, xv, y, yv, t, n):
        if t[n] == 0:
            time.sleep(2)
        if plot_method == 1:
            plt.mesh(x, y, u, title='t=%g' % t[n], zlim=[-1,1],
                 caxis=[-1,1])
        elif plot_method == 2:
            surfc(xv, yv, u, title='t=%g' % t[n], zlim=[-1, 1],
                  colorbar=True, colormap=hot(), caxis=[-1,1],
                  shading='flat')
        elif plot_method == 3:
            print ('Experimental 3D matplotlib...under development...')
            #plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            u_surf = ax.plot_surface(xv, yv, u, alpha=0.3)
            #ax.contourf(xv, yv, u, zdir='z', offset=-100, cmap=cm.coolwarm)
            #ax.set_zlim(-1, 1)
            # Remove old surface before drawing
            if u_surf is not None:
                ax.collections.remove(u_surf)
            plt.draw()
            time.sleep(1)
        if plot_method > 0:
            time.sleep(0) # pause between frames
            if save_plot:
                filename = 'tmp_%04d.png' % n
                plt.savefig(filename)  # time consuming!

    Nx = 40; Ny = 40; T = 20
    dt, cpu = solver(I, None, None, c, Lx, Ly, Nx, Ny, -1, T,
                     user_action=plot_u, version=version)

def test_gaussian():
	for filename in glob.glob('tmp_*.png'):
		os.remove(filename)
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	import matplotlib.pyplot as plt

	Lx = 10
	Ly = 10
	I = lambda x,y: exp(-((x-0/2.0)/2)**2)
	V = None
	f = 0
	b = 0.0; T = 2; Nx = 20; Ny = 20; dt = -1
	c = lambda x,y: 20#sqrt(9.81*(1 - 0.5*exp(-(x-Lx/2)**2 - (y-Ly/2)**2)))   # q = g*H(x,y)= g*(H0 - B(x,y))

	def plot(u_num, x, xv, y, yv, t, n):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X = x
		Y = y
		X, Y = meshgrid(X, Y)
		Z = u_num
		Z2 = -1 + 0.5*exp(-(X-Lx/2)**2 - (Y-Ly/2)**2)
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
				       linewidth=0, antialiased=False)
		ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
		#cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
		cset = ax.contour(X, Y, Z, zdir='x', offset=-1, cmap=cm.coolwarm)
		cset = ax.contour(X, Y, Z, zdir='y', offset=11, cmap=cm.coolwarm)

		ax.set_zlim(-1.01, 1.01)
		ax.set_xlabel('X')
		ax.set_xlim(-1, 10)
		ax.set_ylabel('Y')
		ax.set_ylim(0, 11)
		ax.set_zlabel('Z')
		ax.set_zlim(-100, 100)

		surf = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.hot,
				       linewidth=0, antialiased=False)
		ax.set_zlim(-1.01, 1.01)

		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		fig.colorbar(surf, shrink=0.5, aspect=5)

		plt.savefig('tmp_%04d.png' % n)
		plt.close()
	solver(I, None, None, c, Lx, Ly, Nx, Ny, -1, T,
                     user_action=plot, version='vectorized')
def test_plug():
	""" pulse function for simulating the propagation of a plug wave,
		where I(x) is constant in some region of the domain and
		zero elsewhere

	"""
	for filename in glob.glob('tmp_*.png'):
		os.remove(filename)
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	import matplotlib.pyplot as plt

	Lx = 10
	Ly = 10

	Ix = lambda x,y: 0 if abs(x-Lx/2.0) > 0.5 else 1
	Iy = lambda x,y: 0 if abs(y-Ly/2.0) > 0.1 and abs(x-Lx/2.0) > 0.1 else 1

    #I = lambda x,y: 0 if abs(x+y-Lx/2-Ly/2) > 0.5 else 1

	V = 0
	f = 0
	b = 0.0; c = 1; T = 12; Nx = 50; Ny = 50; dt = 0.05

	def plot(u_num, x, xv, y, yv, t, n):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X = x
		Y = y
		X, Y = meshgrid(X, Y)
		Z = u_num
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
				       linewidth=0, antialiased=False)
		ax.set_zlim(-1.01, 1.01)

		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		fig.colorbar(surf, shrink=0.5, aspect=5)

		plt.savefig('tmp_%04d.png' % n)
		plt.close()
	solver(Iy,V,f,c,Lx,Ly,Nx,Ny,dt,T,b,lambda x,y: c,
		  user_action=plot, version='vectorized')
def plot(u_num, x, xv, y, yv, t, n):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = x
	Y = y
	X, Y = meshgrid(X, Y)
	Z = u_num
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
				       linewidth=0, antialiased=False)
	ax.set_zlim(-1.01, 1.01)

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.savefig('tmp_%04d.png' % n)
	plt.close()
def test_standing_undamped_waves(u, q):
    """Exact discrete solution of the scheme."""
    """
    The convergence rate will note go towards 2. After a lot of troubleshooting, we
    do not know why.
    """
    A = 1
    B = 1
    omega =1
    Lx = 10
    Ly = 10
    c = 1.0
    T = 18
    b = 0
    f = 0
    Nx = 50
    Ny = 50
    dt = 1e-1


    x,y,t = sym.symbols("x y t")


    u_t = sym.diff(u,t)


    u_tt = sym.diff(u_t,t)
    u_x = sym.diff(u,x)

    q_ux = q*u_x
    q_ux_x = sym.diff(q_ux,x)

    u_y = sym.diff(u,y)
    q_uy = q*u_y
    q_uy_y = sym.diff(q_uy,y)

    f = u_tt-q_ux_x-q_uy_y

    f = sym.lambdify((x,y,t),f)
    V = sym.lambdify((x,y),u_t.subs(t,0))
    I = sym.lambdify((x,y),u.subs(t,0))

    dt_values = [dt*2**(-i) for i in range(5)]
    q = sym.lambdify((x,y),q)

    u_exact = sym.lambdify((x,y,t),u)

    def assert_no_error(u, x, xv, y, yv, t, n):

        u_e = u_exact(xv,yv,t[n])
        diff = abs(u - u_e).max()
        tol = 4
        msg = 'diff=%g, step %d, time=%g' % (diff, n, t[n])
        assert diff < tol, msg
    E_value = zeros(5)

    def compute_error(u_num,x,xv,y,yv,t,n):
        E = abs(u_exact(x,y,t[n])-u_num).max()

        for i in range(len(dt_values)):
            if t[n]-t[n-1] == dt_values[i]:
                if E > E_value[i]:
                    E_value[i] = E

    def convergence_rate(E,h):
        m = len(dt_values)
        r = [log(E[i]/E[i-1])/log(h[i]/h[i-1]) for i in range(1,m,1)]
        #r = [round(r_,2) for r_ in r]
        return r

    for _dt in dt_values:
        new_dt, cpu = solver(I,V,f,c,Lx,Ly,Nx,Ny,_dt, T,b,q,
        user_action=compute_error,version='vectorized')

    print ('Values of norm error =', E_value)
    print ('convergence_rate=',convergence_rate(E_value, dt_values))
    """

	def convergence_rate(E, h):
		m = len(dt_values)
		r = [log(E[i]/E[i-1])/log(h[i]/h[i-1]) for i in range(1,m, 1)]
		r = [round(r_,2) for r_ in r]
		return r

    for _dt in dt_values:
        new_dt, cpu = solver(I, V, f, c, Lx, Ly, Nx, Ny, _dt, T,b,q,
        user_action=assert_no_error, version='vectorized')


	print 'Error norm values = ', E_value
	print convergence_rate(E_value, dt_values)
    """

    return new_dt, cpu


if __name__ == '__main__':


    test_quadratic()
    test_plug()

    A = 1; B=1;
    omega =1
    Lx = 10
    Ly = 10
    c = 1.0
    T = 18
    b = 0
    f = 0
    Nx = 50
    Ny = 50
    dt = 1e-1

    x,y,t,q = sym.symbols("x y t q")
    #takes u and q as input
    test_standing_undamped_waves(A*sym.cos(sym.pi*x/Lx)*sym.cos(sym.pi*y/Ly)*sym.cos(omega*t), c)
    test_standing_undamped_waves((A*sym.cos(omega*t)+B*sym.sin(omega*t))*sym.exp(-c*t)*sym.cos(sym.pi*x/Lx)*sym.cos(sym.pi*y/Ly),x**2)
