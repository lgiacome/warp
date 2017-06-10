"""Classes following the PICMI standard
"""
import PICMI_Base
from warp import *
import warp


class Grid(PICMI_Base.PICMI_Grid):

    def init(self, **kw):
        w3d.nx = self.nx
        w3d.ny = self.ny
        w3d.nz = self.nz
        w3d.xmmin = self.xmin
        w3d.xmmax = self.xmax
        w3d.ymmin = self.ymin
        w3d.ymmax = self.ymax
        w3d.zmmin = self.zmin
        w3d.zmmax = self.zmax

        defaultsdict = {'dirichlet':dirichlet,
                        'neumann':neumann,
                        'periodic':periodic,
                        'open':openbc}
        self.bounds = [defaultsdict[self.bcxmin], defaultsdict[self.bcxmax],
                       defaultsdict[self.bcymin], defaultsdict[self.bcymax],
                       defaultsdict[self.bczmin], defaultsdict[self.bczmax]]
        w3d.boundxy = self.bounds[1]
        w3d.bound0 = self.bounds[4]
        w3d.boundnz = self.bounds[5]


class EM_solver(PICMI_Base.PICMI_EM_solver):

    def init(self, **kw):

        self.solver = EM3D()
    

class Species(warp.Species):

    def add_particles(self, n=None,
                      x=None, y=None, z=None,
                      ux=None, uy=None, uz=None, w=None,
                      unique_particles=None, **kw):
        return self.addparticles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w, unique_particles=unique_particles)


class Simulation(PICMI_Base.PICMI_Simulation):
    def init(self, **kw):
        if self.verbose is not None:
            top.verbosity = self.verbose + 1

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

        
