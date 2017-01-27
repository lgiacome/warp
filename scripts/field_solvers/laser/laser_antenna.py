import numpy as np
from ...warp import PicklableFunction

class LaserAntenna(object):

    def __init__(self, laser_func,vector, polvector, spot, emax,
                 source_z, source_v, polangle, w3d, dim, circ_m ):

        # Initialize the variable self.spot
        if spot is None:
            if source_z is None:
                source_z = w3d.zmmin
            source_z = max(min(source_z,w3d.zmmax),w3d.zmmin)
            self.spot = np.array([ 0, 0, source_z ])
        else:
            self.spot = spot

        # Initialize the variable self.polvector
        if polvector is None:
            if polangle is None:
                polangle = 0.
            polvector = np.array([ np.cos(polangle), np.sin(polangle), 0.] )

        # Error if the 2 main vectors are not orthognal
        assert np.isclose(np.dot(vector, polvector), 0.), \
            "Error : laser_vector and laser_polvector must be orthogonal. "

        # Normalisation of vector and polvector
        vect_norm = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        polvect_norm = np.sqrt(polvector[0]**2+polvector[1]**2+polvector[2]**2)

        self.vector = vector/vect_norm
        self.polvector = polvector/polvect_norm

        # Creation of a third vector orthogonal to both vector and polvector
        self.polvector_2 = np.cross(self.vector, self.polvector)

        #
        self.emax = emax

        #
        self.laser_func = PicklableFunction( laser_func )

        # Dimension
        self.circ_m = circ_m
        self.dim = dim

        # Particle initialization
        self.initialize_virtual_particles( w3d )

        # Antenna velocity
        self.v = source_v
        self.vx = self.v * self.vector[0] * np.ones(self.nn)
        self.vy = self.v * self.vector[1] * np.ones(self.nn)
        self.vz = self.v * self.vector[2] * np.ones(self.nn)

    def initialize_virtual_particles( self, w3d ):
        """
        Initialization of the antenna particles depending on the dimension and
        the laser propagation vector.

        """

        # Shortcut definition
        x0 = self.spot[0]
        y0 = self.spot[1]
        z0 = self.spot[2]

        xmin = w3d.xmminlocal
        xmax = w3d.xmmaxlocal
        ymin = w3d.ymminlocal
        ymax = w3d.ymmaxlocal
        zmin = w3d.zmminlocal
        zmax = w3d.zmmaxlocal

        if self.dim == "1d":
            # 1D injection along x
            self.nn = 1
            self.xx = x0 + np.zeros(self.nn)
            self.yy = y0 + np.zeros(self.nn)
            self.zz = z0 + np.zeros(self.nn)

        elif self.dim == "circ":
            #2D circ
            #CORRECTIONS NEEDED : rr, nlas etc..
            rr = self.xx.copy()
            self.weights_circ=2*np.pi*rr/w3d.dx/nlas
            self.weights_circ/=4*self.circ_m
            w0 = self.weights_circ.copy()
            for i in range(1,4*self.circ_m):
                phase = 0.5*np.pi*float(i)/self.circ_m
                self.xx = np.concatenate((self.xx,rr*cos(phase)))
                self.yy = np.concatenate((self.yy,rr*sin(phase)))
                self.weights_circ = np.concatenate((self.weights_circ,w0))
            self.nn = np.shape(self.xx)[0]
            self.zz = z0 + np.zeros(self.nn)

        elif self.dim == "2d":
            # 2D plane

            # Ux is chosen orthogonal to vector in the plane (x,z)
            Uy = np.array([0.,1.,0.])
            Ux = np.cross(Uy,self.vector)

            # Spacing between virtual particles to ensure one particle per cell
            # select only the Ux components different from 0
            list_Ux = []
            if not Ux[0] == 0.: list_Ux.append(w3d.dx/np.abs(Ux[0]))
            if not Ux[1] == 0.: list_Ux.append(w3d.dy/np.abs(Ux[1]))
            if not Ux[2] == 0.: list_Ux.append(w3d.dz/np.abs(Ux[2]))
            self.Sx = min(list_Ux)

            xmin_i = self.switch_min_max(xmin, xmax, Ux[0])
            ymin_i = self.switch_min_max(ymin, ymax, Ux[1])
            zmin_i = self.switch_min_max(zmin, zmax, Ux[2])
            xmax_i = self.switch_min_max(xmax, xmin, Ux[0])
            ymax_i = self.switch_min_max(ymax, ymin, Ux[1])
            zmax_i = self.switch_min_max(zmax, zmin, Ux[2])

            imin = Ux[0]*(xmin_i-x0) + Ux[1]*(ymin_i-y0) + Ux[2]*(zmin_i-z0)
            imax = Ux[0]*(xmax_i-x0) + Ux[1]*(ymax_i-y0) + Ux[2]*(zmax_i-z0)

            imin = np.floor(imin/self.Sx)
            imax = np.floor(imax/self.Sx)+1

            antenna_i = np.arange(imin, imax)
            self.xx = x0 + self.Sx*Ux[0]*antenna_i
            self.zz = z0 + self.Sx*Ux[2]*antenna_i

            # Normalization to respect the boundaries
            self.zz = self.boundaries_reduction(self.zz,self.xx, xmin, xmax)
            self.xx = self.boundaries_reduction(self.xx,self.xx, xmin, xmax)

            self.xx = self.boundaries_reduction(self.xx,self.zz, zmin, zmax)
            self.zz = self.boundaries_reduction(self.zz,self.zz, zmin, zmax)

            self.yy = np.zeros(len(self.xx))
            self.Ux = Ux
            self.Uy = Uy
            # Number of virtual particles
            self.nn = np.shape(self.xx)[0]

        else:
            # 3D case, Ux = polvector and Uy = polvector_2
            Ux = self.polvector
            Uy = self.polvector_2

            # Spacing between virtual particles to ensure one particle per cell
            # select only the components of Ux and Uy different from 0
            list_Ux = []; list_Uy = []
            if not Ux[0] == 0.: list_Ux.append( w3d.dx/np.abs(Ux[0]) )
            if not Ux[1] == 0.: list_Ux.append( w3d.dy/np.abs(Ux[1]) )
            if not Ux[2] == 0.: list_Ux.append( w3d.dz/np.abs(Ux[2]) )
            if not Uy[0] == 0.: list_Uy.append( w3d.dx/np.abs(Uy[0]) )
            if not Uy[1] == 0.: list_Uy.append( w3d.dy/np.abs(Uy[1]) )
            if not Uy[2] == 0.: list_Uy.append( w3d.dz/np.abs(Uy[2]) )
            self.Sx = min(list_Ux)
            self.Sy = min(list_Uy)

            xmin_i = self.switch_min_max(xmin, xmax, Ux[0])
            ymin_i = self.switch_min_max(ymin, ymax, Ux[1])
            zmin_i = self.switch_min_max(zmin, zmax, Ux[2])
            xmax_i = self.switch_min_max(xmax, xmin, Ux[0])
            ymax_i = self.switch_min_max(ymax, ymin, Ux[1])
            zmax_i = self.switch_min_max(zmax, zmin, Ux[2])

            xmin_j = self.switch_min_max(xmin, xmax, Uy[0])
            ymin_j = self.switch_min_max(ymin, ymax, Uy[1])
            zmin_j = self.switch_min_max(zmin, zmax, Uy[2])
            xmax_j = self.switch_min_max(xmax, xmin, Uy[0])
            ymax_j = self.switch_min_max(ymax, ymin, Uy[1])
            zmax_j = self.switch_min_max(zmax, zmin, Uy[2])

            imin = Ux[0]*(xmin_i-x0) + Ux[1]*(ymin_i-y0) + Ux[2]*(zmin_i-z0)
            imax = Ux[0]*(xmax_i-x0) + Ux[1]*(ymax_i-y0) + Ux[2]*(zmax_i-z0)
            jmin = Uy[0]*(xmin_j-x0) + Uy[1]*(ymin_j-y0) + Uy[2]*(zmin_j-z0)
            jmax = Uy[0]*(xmax_j-x0) + Uy[1]*(ymax_j-y0) + Uy[2]*(zmax_j-z0)

            imin = np.floor(imin/self.Sx)
            imax = np.floor(imax/self.Sx)+1
            jmin = np.floor(jmin/self.Sy)
            jmax = np.floor(jmax/self.Sy)+1

            array_i = np.arange(imin, imax)
            array_j = np.arange(jmin, max)
            antenna_i, antenna_j = np.meshgrid(array_i,array_j)

            self.xx = x0 + self.Sx*Ux[0]*antenna_i + self.Sy*Uy[0]*antenna_j
            self.yy = y0 + self.Sx*Ux[1]*antenna_i + self.Sy*Uy[1]*antenna_j
            self.zz = z0 + self.Sx*Ux[2]*antenna_i + self.Sy*Uy[2]*antenna_j

            self.xx = self.xx.flatten()
            self.yy = self.yy.flatten()
            self.zz = self.zz.flatten()

            # Normalization to respect the boundaries
            self.yy = self.boundaries_reduction(self.yy,self.xx, xmin, xmax)
            self.zz = self.boundaries_reduction(self.zz,self.xx, xmin, xmax)
            self.xx = self.boundaries_reduction(self.xx,self.xx, xmin, xmax)

            self.xx = self.boundaries_reduction(self.xx,self.yy, ymin, ymax)
            self.zz = self.boundaries_reduction(self.zz,self.yy, ymin, ymax)
            self.yy = self.boundaries_reduction(self.yy,self.yy, ymin, ymax)

            self.xx = self.boundaries_reduction(self.xx,self.zz, zmin, zmax)
            self.yy = self.boundaries_reduction(self.yy,self.zz, zmin, zmax)
            self.zz = self.boundaries_reduction(self.zz,self.zz, zmin, zmax)

            self.Ux = Ux
            self.Uy = Uy
            # Number of virtual particles
            self.nn = np.shape(self.xx)[0]

        # Set the deplacement around the initial position and normalized momenta
        # variation of each macroparticles to 0
        self.xdx = np.zeros(self.nn)
        self.ydy = np.zeros(self.nn)
        self.zdz = np.zeros(self.nn)

        self.ux = np.zeros(self.nn)
        self.uy = np.zeros(self.nn)
        self.uz = np.zeros(self.nn)

        self.gi = np.ones(self.nn)

    def push_virtual_particles(self, top, f, clight, eps0):
        """
        Calculate the motion parameters of the laser antenna at a given
        timestep
        """
        x0 = self.spot[0]
        y0 = self.spot[1]
        z0 = self.spot[2]
        dt = top.dt

        Ux = self.Ux
        Uy = self.Uy

        self.xx += self.vx * dt
        self.yy += self.vy * dt
        self.zz += self.vz * dt

        #Coordinate of the antenna in the plane (Ux,Uy)
        x = (self.xx-x0)*Ux[0] + (self.yy-y0)*Ux[1] + (self.zz-z0)*Ux[2]
        y = (self.xx-x0)*Uy[0] + (self.yy-y0)*Uy[1] + (self.zz-z0)*Uy[2]
        t = top.time*(1.-self.v/clight)
        amp = self.laser_func(x,y,t)

        # --- displaces fixed weight particles on "continuous" trajectories
        dispmax = 0.01*clight
        coef_ampli = dispmax * (1.-self.v/clight) / self.emax

        if isinstance(amp,list): #elliptic polarization
            amp_x = amp[0]*self.polvector[0] + amp[1]*self.polvector_2[0]
            amp_y = amp[0]*self.polvector[1] + amp[1]*self.polvector_2[2]
            amp_z = amp[0]*self.polvector[2] + amp[1]*self.polvector_2[2]

            amplitude_x = coef_ampli * amp_x
            amplitude_y = coef_ampli * amp_y
            amplitude_z = coef_ampli * amp_z

        else: #linear polarization
            amplitude_x = coef_ampli * amp * self.polvector[0]
            amplitude_y = coef_ampli * amp * self.polvector[1]
            amplitude_z = coef_ampli * amp * self.polvector[2]

        # Set the amplitude of the normalized momenta of the fictious
        # macroparticles
        self.ux[...] = amplitude_x
        self.uy[...] = amplitude_y
        self.uz[...] = amplitude_z

        # Set the corresponding displacement of the fictious macroparticles
        self.xdx[...] += self.ux * dt
        self.ydy[...] += self.uy * dt
        self.zdz[...] += self.uz * dt

        self.weights = np.ones(self.nn) * eps0 * self.emax/0.01

        if not self.dim == "1d": # 2D and 3D
            self.weights *= self.Sx
        if self.dim == "3d" : # 3D cartesian
            self.weights *= self.Sy
        if self.circ_m > 0 : # Circ
            # Laser initialized with particles in a star-pattern
            self.weights *= f.dx * self.weights_circ


    def depose_source(self, func, f, q, dt, zgrid, order_x, order_y, order_z):
        """
        Depose the source that generates the laser, on the grid (generation of
        the laser by an antenna).

        Notice that this current is generated by fictious macroparticles, whose
        motion is explicitly specified. Thus, this current does not correspond
        to that of the actual macroparticles of the simulation.
        """
        l_particles_weight= True

        func(self.nn,
             f,
             self.xx + q*self.xdx,
             self.yy + q*self.ydy,
             self.zz + q*self.zdz,
             self.vx + q*self.ux,
             self.vy + q*self.uy,
             self.vz + q*self.uz,
             self.gi,
             dt,
             self.weights,
             zgrid,
             q,
             1.,
             order_x,
             order_y,
             order_z,
             l_particles_weight)


    #Additional routines
    def switch_min_max(self,x1,x2,u):
        """
            Return x1 or x2 depending on the sign of u
        """
        if u >= 0 :
            return x1
        else:
            return x2

    def boundaries_reduction(self, u, x, xmin, xmax):
        """
            u and x, two same size vectors
            Return the values of u such as xmin <= x < xmax
        """
        u = u[x >= xmin]; x = x[x >= xmin]
        u = u[x < xmax]
        return u
