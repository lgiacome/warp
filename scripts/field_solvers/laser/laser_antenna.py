import numpy as np
from warp import PicklableFunction

class LaserAntenna(object):


    def __init__(self, laser_func,
                vector, polvector, spot, emax, source_z, polangle, w3d
                 dim, circ_m ):

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

        #
        self.laser_func = PicklableFunction( laser_func )

        #Normalisation of vector and polvector
        self.vector = vector/np.sqrt(vector[0]**2 + vector[1]**2+ vector[2]**2)
        self.polvector = polvector/np.sqrt(polvector[0]**2+polvector[1]**2+polvector[2]**2)
        # Creation of a third vector orthogonal to both vector and polvector
        self.polvector_2 = cross(self.vector, self.polvector)


        #
        self.circ_m = circ_m
        self.dim = dim

        self.initialize_virtual_macroparticles()

    def initialize_virtual_macroparticles( self, w3d ):
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
            #CORRECTIONS NEEDED : rr, nlas etc..
            rr = self.xx.copy()
            self.weights_circ=2*np.pi*rr/w3d.dx/nlas
            self.weights_circ/=4*self.circ_m
            w0 = self.weights_circ.copy()
            for i in range(1,4*self.circ_m):
                self.xx = np.concatenate((self.xx,rr*cos(0.5*np.pi*float(i)/self.circ_m)))
                self.yy = np.concatenate((self.yy,rr*sin(0.5*np.pi*float(i)/self.circ_m)))
                self.weights_circ = np.concatenate((self.weights_circ,w0))
            self.nn = np.shape(self.xx)[0]
            self.zz = z0 + np.zeros(self.nn)

        elif self.dim == "2d":
            # 2D with Ux orthogonal to vector in the plane (x,z)
            self.Uy = np.array([0.,1.,0.])
            self.Ux = np.cross(self.Uy,self.vector)

            # Spacing between virtual particles to ensure one particle per cell
            # select only the components of Ux different from 0
            list_Ux = []
            if not self.Ux[0]==0.: list_Ux.append(w3d.dx/np.abs(self.Ux[0]))
            if not self.Ux[1]==0.: list_Ux.append(w3d.dy/np.abs(self.Ux[1]))
            if not self.Ux[2]==0.: list_Ux.append(w3d.dz/np.abs(self.Ux[2]))
            self.Sx = min(list_Ux)

            xmin_i = self.switch_min_max(xmin, xmax, self.Ux[0])
            ymin_i = self.switch_min_max(ymin, ymax, self.Ux[1])
            zmin_i = self.switch_min_max(zmin, zmax, self.Ux[2])
            xmax_i = self.switch_min_max(xmax, xmin, self.Ux[0])
            ymax_i = self.switch_min_max(ymax, ymin, self.Ux[1])
            zmax_i = self.switch_min_max(zmax, zmin, self.Ux[2])

            imin = self.Ux[0]*(xmin_i-x0)+self.Ux[1]*(ymin_i-y0)+self.Ux[2]*(zmin_i-z0)
            imax = self.Ux[0]*(xmax_i-x0)+self.Ux[1]*(ymax_i-y0)+self.Ux[2]*(zmax_i-z0)

            imin = np.floor(antenna_imin/self.Sx)
            imax = np.floor(antenna_imax/self.Sx)+1

            self.antenna_i = np.arange(imin, imax)
            self.xx = x0 + self.Sx*self.Ux[0]*self.antenna_i
            self.zz = z0 + self.Sx*self.Ux[2]*self.antenna_i

            # Normalization to respect the boundaries
            self.zz = self.boundaries_reduction(self.zz,self.xx, xmin, xmax)
            self.xx = self.boundaries_reduction(self.xx,self.xx, xmin, xmax)

            self.xx = self.boundaries_reduction(self.xx,self.zz, zmin, zmax)
            self.zz = self.boundaries_reduction(self.zz,self.zz, zmin, zmax)

            self.yy = np.zeros(len(self.xx))
            # Number of virtual particles
            self.nn = np.shape(self.xx)[0]

        else:
            # 3D case, Ux = polvector and Uy = polvector_2
            self.Ux = self.polvector
            self.Uy = self.polvector_2

            # Spacing between virtual particles to ensure one particle per cell
            # select only the components of Ux and Uy different from 0
            list_Ux = []; list_Uy = []
            if not self.Ux[0]==0.: list_Ux.append(f.dx/np.abs(self.Ux[0]))
            if not self.Ux[1]==0.: list_Ux.append(f.dy/np.abs(self.Ux[1]))
            if not self.Ux[2]==0.: list_Ux.append(f.dz/np.abs(self.Ux[2]))
            if not self.Uy[0]==0.: list_Uy.append(f.dx/np.abs(self.Uy[0]))
            if not self.Uy[1]==0.: list_Uy.append(f.dy/np.abs(self.Uy[1]))
            if not self.Uy[2]==0.: list_Uy.append(f.dz/np.abs(self.Uy[2]))
            self.Sx = min(list_Ux)
            self.Sy = min(list_Uy)

            xmin_i = self.switch_min_max(xmin, xmax, self.Ux[0])
            ymin_i = self.switch_min_max(ymin, ymax, self.Ux[1])
            zmin_i = self.switch_min_max(zmin, zmax, self.Ux[2])
            xmax_i = self.switch_min_max(xmax, xmin, self.Ux[0])
            ymax_i = self.switch_min_max(ymax, ymin, self.Ux[1])
            zmax_i = self.switch_min_max(zmax, zmin, self.Ux[2])

            xmin_j = self.switch_min_max(xmin, xmax, self.Uy[0])
            ymin_j = self.switch_min_max(ymin, ymax, self.Uy[1])
            zmin_j = self.switch_min_max(zmin, zmax, self.Uy[2])
            xmax_j = self.switch_min_max(xmax, xmin, self.Uy[0])
            ymax_j = self.switch_min_max(ymax, ymin, self.Uy[1])
            zmax_j = self.switch_min_max(zmax, zmin, self.Uy[2])

            imin = self.Ux[0]*(xmin_i-x0)+self.Ux[1]*(ymin_i-y0)+self.Ux[2]*(zmin_i-z0)
            imax = self.Ux[0]*(xmax_i-x0)+self.Ux[1]*(ymax_i-y0)+self.Ux[2]*(zmax_i-z0)
            jmin = self.Uy[0]*(xmin_j-x0)+self.Uy[1]*(ymin_j-y0)+self.Uy[2]*(zmin_j-z0)
            jmax = self.Uy[0]*(xmax_j-x0)+self.Uy[1]*(ymax_j-y0)+self.Uy[2]*(zmax_j-z0)

            imin = np.floor(imin/self.Sx)
            imax = np.floor(imax/self.Sx)+1
            jmin = np.floor(jmin/self.Sy)
            jmax = np.floor(jmax/self.Sy)+1

            array_i = np.arange(imin, imax)
            array_j = np.arange(jmin, max)
            self.antenna_i, self.antenna_j = np.meshgrid(array_i,array_j)

            self.xx = x0 + self.Sx*self.Ux[0]*self.antenna_i + self.Sy*self.Uy[0]*self.antenna_j
            self.yy = y0 + self.Sx*self.Ux[1]*self.antenna_i + self.Sy*self.Uy[1]*self.antenna_j
            self.zz = z0 + self.Sx*self.Ux[2]*self.antenna_i + self.Sy*self.Uy[2]*self.antenna_j

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

            # Number of virtual particles
            self.nn = np.shape(self.xx)[0]

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
            u=  u[x >= xmin]; x = x[x >= xmin]
            u = u[x < xmax]
            return u
