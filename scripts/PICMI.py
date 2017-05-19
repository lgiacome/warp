"""Classes following the PICMI standard
"""
from warp import *
import warp


class Grid(object):
    """
    - `Grid`
      - **type**: *object*
      - `Nx=nx` - **type**: *integer* - "Number of cells along X (Nb nodes=nx+1)."
      - `Ny=ny` - **type**: *integer* - "Number of cells along Y (Nb nodes=ny+1)."
      - `Nr=nr` - **type**: *integer* - "Number of cells along R (Nb nodes=nr+1)."
      - `Nz=nz` - **type**: *integer* - "Number of cells along Z (Nb nodes=nz+1)."
      - `Nm=nm` - **type**: *integer* - "Number of azimuthal modes."
      - `Xmin=xmin` - **type**: *double* - "Position of first node along X."
      - `Xmax=xmax` - **type**: *double* - "Position of last node along X."
      - `Ymin=ymin` - **type**: *double* - "Position of first node along Y."
      - `Ymax=ymax` - **type**: *double* - "Position of last node along Y."
      - `Rmax=rmax` - **type**: *double* - "Position of last node along R."
      - `Zmin=zmin` - **type**: *double* - "Position of first node along Z."
      - `Zmax=zmax` - **type**: *double* - "Position of last node along Z."
      - `bcxmin` - **type**: *string* - "Boundary condition at min X: periodic/open/dirichlet/neumann."
      - `bcxmax` - **type**: *string* - "Boundary condition at max X: periodic/open/dirichlet/neumann."
      - `bcymin` - **type**: *string* - "Boundary condition at min Y: periodic/open/dirichlet/neumann."
      - `bcymax` - **type**: *string* - "Boundary condition at max Y: periodic/open/dirichlet/neumann."
      - `bcrmax` - **type**: *string* - "Boundary condition at max R: open/dirichlet/neumann."
      - `bczmin` - **type**: *string* - "Boundary condition at min Z: periodic/open/dirichlet/neumann."
      - `bczmax` - **type**: *string* - "Boundary condition at max Z: periodic/open/dirichlet/neumann."

    """

    def __init__(self, nx = None, ny = None, nr = None, nz = None, nm = None,
                 xmin = None, xmax = None, ymin = None, ymax = None, rmax = None, zmin = None, zmax = None,
                 bcxmin = None, bcxmax = None, bcymin = None, bcymax = None, bcrmax = None, bczmin = None, bczmax = None,
                 **kw):
        self.nx = nx
        self.ny = ny
        self.nr = nr
        self.nz = nz
        self.nm = nm
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax
        self.bcxmin = bcxmin
        self.bcxmax = bcxmax
        self.bcymin = bcymin
        self.bcymax = bcymax
        self.bcrmax = bcrmax
        self.bczmin = bczmin
        self.bczmax = bczmax

        w3d.nx = nx
        w3d.ny = ny
        w3d.nz = nz
        w3d.xmmin = xmin
        w3d.xmmax = xmax
        w3d.ymmin = ymin
        w3d.ymmax = ymax
        w3d.zmmin = zmin
        w3d.zmmax = zmax

        defaultsdict = {'dirichlet':dirichlet,
                        'neumann':neumann,
                        'periodic':periodic,
                        'open':openbc}
        self.bounds = [defaultsdict[bcxmin], defaultsdict[bcxmax],
                       defaultsdict[bcymin], defaultsdict[bcymax],
                       defaultsdict[bczmin], defaultsdict[bczmax]]
        w3d.boundxy = self.bounds[1]
        w3d.bound0 = self.bounds[4]
        w3d.boundnz = self.bounds[5]


class EM_solver(object):
    Methods_list = ['Yee', 'CK', 'CKC', 'Lehe', 'PSTD', 'PSATD', 'GPSTD']
    def __init__(self, Method=None,
                 norderx=None, nordery=None, norderr=None, norderz=None,
                 l_nodal=None,
                 current_deposition_algo=None, charge_deposition_algo=None,
                 field_gathering_algo=None, particle_pusher_algo=None, **kw):

        assert Method is None or Method in EM_solver.Methods_list, Exception('Method has incorrect value')

        self.solver = EM3D()
    

class Species(warp.Species):
    def __init__(self,
                  Type=None, type=None,
                  Name=None, name=None,
                  Sid=None, sid=None,
                  Charge_state=None, charge_state=None,
                  Charge=None, charge=None, Q=None, q=None,
                  Mass=None, mass=None, M=None, m=None,
                  Weight=None, weight=1., W=None, w=None, **kw):
        # --- Accept multpiple names, but use 'type', 'name', 'sid', 'charge_state', 'charge', 'mass', 'weight'
        if Type is not None: type = Type
        if Name is not None: name = Name
        if Sid is not None: sid = Sid
        if Charge_state is not None: charge_state = Charge_state
        if Charge is not None: charge = Charge
        if Q is not None: charge = Q
        if q is not None: charge = q
        if Mass is not None: mass = Mass
        if M is not None: mass = M
        if m is not None: mass = m
        if Weight is not None: weight = Weight
        if W is not None: weight = W
        if w is not None: weight = w

        warp.Species.__init__(self, type=type, charge=charge, mass=mass,
                              charge_state=charge_state, weight=weight, name=name, **kw)

    def add_particles(self, n=None,
                      x=None, y=None, z=None,
                      ux=None, uy=None, uz=None, w=None,
                      unique_particles=None, **kw):
        return self.addparticles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w, unique_particles=unique_particles)


class Simulation(object):
    def __init__(self, plot_int=None, verbose=None, cfl=None):
        self.plot_int = plot_int
        self.verbose = verbose

        if verbose is not None:
            top.verbosity = verbose + 1

        package('w3d')
        generate()

        installafterstep(self.makedumps)

    def makedumps(self):
        if self.plot_int is not None and self.plot_int > 0:
            if top.it%self.plot_int == 0:
                dump()

    def step(self, nsteps=1):
        step(nsteps)
        
    def finalize(self):
        uninstallafterstep(self.makedumps)

        
