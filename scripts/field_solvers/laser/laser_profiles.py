"""
This file defines a set of standard laser profiles, that can be passed
as `laser_func` to the EM3D class or to the `LaserAntenna` class
"""

import numpy as np
from scipy.constants import c, m_e, e
from scipy.interpolate import RegularGridInterpolator
from scipy.special import genlaguerre
#from .boost_tools import BoostConverter
import h5py
# Try importing parallel functions, in order to broadcast
# the experimental laser file, if required
try:
    from warp.parallel import mpibcast, me
except ImportError:
    # Single-proc simulation
    mpibcast = lambda x:x
    me = 0

class ExperimentalProfile( object ):
    """Class that calculates the laser from a data file."""

    def __init__( self, k0, laser_file, laser_file_energy ):

        # The first processor loads the file and sends it to the others
        # (This prevents all the processors from accessing the same file,
        # which can substantially slow down the simulation)
        if me==0:
            with h5py.File(laser_file) as f:
                x = f['x'][:]
                y = f['y'][:]
                t = f['t'][:]
                Ereal = f['Ereal'][:,:]
                Eimag = f['Eimag'][:,:]
        else:
            r = None
            t = None
            Ereal = None
            Eimag = None
        # Broadcast the data to all procs
        r = mpibcast( r )
        t = mpibcast( t )
        Ereal = mpibcast( Ereal )
        Eimag = mpibcast( Eimag )

        # Recover the complex field
        E_data = Ereal + 1.j*Eimag

        # Change the value of the field (by default it is 1J)
        E_norm = np.sqrt( laser_file_energy )
        E_data = E_data*E_norm
        self.E0 = abs(E_data).max()

        # Register the wavevector
        self.k0 = k0

        # Interpolation object
        self.interp_func = RegularGridInterpolator( (t, r), E_data,
                            bounds_error=False, fill_value=0. )

    def __call__( self, x, y, t ):
        """
        Return the transverse profile of the laser at the position
        of the antenna

        Parameters:
        -----------
        x: float or ndarray
            First transverse direction in meters

        y: float or ndarray
            Second transverse direction in meters

        t: float
            Time in seconds
        """
        # Calculate the array of radius
        r = np.sqrt( x**2 + y**2 )

        # Interpolate to find the complex amplitude
        Ecomplex = self.interp_func( (t, r) )

        # Add laser oscillations
        Eosc = ( Ecomplex * np.exp( -1.j*self.k0*c*t ) ).real

        return( Eosc )

class GaussianProfile( object ):
    """Class that calculates a Gaussian laser pulse."""

    def __init__( self, k0, waist, tau, t_peak, focal_length, source_v, a0,
                  dim, boost, temporal_order = 2 ):

        # Set a number of parameters for the laser
        E0 = a0*m_e*c**2*k0/e
        zr = 0.5*k0*waist**2
        # Set default focusing position
        if focal_length is None : focal_length = - c * t_peak

        # Store the parameters
        self.k0 = k0
        self.inv_waist2 = 1./waist**2
        self.inv_zr = 1./zr
        self.inv_tau = 1./tau
        self.t_peak = t_peak
        self.E0 = E0
        self.v_antenna = source_v
        self.focal_length = focal_length
        self.boost = boost
        self.temporal_order = temporal_order

        # Geometric coefficient (for the evolution of the amplitude)
        # In 1D, there is no transverse components, therefore the geomtric
        # coefficient shouldn't be included.
        if  dim=="1d":
            self.geom_coeff = 0.
        elif dim=="2d":
            self.geom_coeff = 0.5
        elif dim in ["circ", "3d"]:
            self.geom_coeff = 1.

    def __call__( self, x, y, t_modified ):
        """
        Return the transverse profile of the laser at the position
        of the antenna

        Parameters:
        -----------
        x: float or ndarray
            First transverse direction in meters

        y: float or ndarray
            Second transverse direction in meters

        t_modified: float
            Time in seconds, multiplied by (1-v_antenna/c)
            This multiplication is done in em3dsolver.py, when
            calling the present function.
        """
        # Calculate the array of radius
        r2 = x**2 + y**2

        # Get the true time
        # (The top.time has been multiplied by (1-v_antenna/c)
        # in em3dsolver.py, before calling the present function)
        t = t_modified/(1.-self.v_antenna/c)

        # Lab-frame formula for the laser:
        # - Waist and curvature and the position of the source
        z =  self.focal_length
        diffract_factor = 1 - 1j*z*self.inv_zr

        # Calculate the argument of the complex exponential
        exp_argument = 1j * self.k0*c*( t - self.t_peak ) \
          - (y**2 + x**2) * self.inv_waist2 / diffract_factor \
          - ( (t - self.t_peak ) * self.inv_tau )**self.temporal_order

        # - Combine profiles
        profile =  np.exp(exp_argument) / ( diffract_factor**self.geom_coeff )

        # Return the combination of profile and field amplitude
        return( self.E0 * profile.real )


class GaussianSTCProfile( object ):
    """Class that calculates a Gaussian laser pulse
    with spatio-temporal correlations (STC)"""

    def __init__( self, k0, waist, ctau, z0, zf, source_z, source_v,
                  a0, zeta, beta, phi2, dim, boost ):

        # Set a number of parameters for the laser
        E0 = a0*m_e*c**2*k0/e
        zr = 0.5*k0*waist**2
        # Set default focusing position
        if zf is None: zf = z0

        # Store the parameters
        self.k0 = k0
        self.inv_zr = 1./zr
        self.inv_waist2 = 1./waist**2
        self.inv_tau2 = c**2/ctau**2
        self.zf = zf
        self.z0 = z0
        self.v_antenna = source_v
        self.source_z = source_z
        self.E0 = E0
        self.beta = beta
        self.zeta = zeta
        self.phi2 = phi2
        self.boost = boost

        # Geometric coefficient (for the evolution of the amplitude)
        # In 1D, there is no transverse components, therefore the geomtric
        # coefficient shouldn't be included.
        if  dim=="1d":
            self.geom_coeff = 0.
        elif dim=="2d":
            self.geom_coeff = 0.5
        elif dim in ["circ", "3d"]:
            self.geom_coeff = 1.


    def __call__( self, x, y, t_modified ):
        """
        Return the transverse profile of the laser at the position
        of the antenna

        Parameters:
        -----------
        x: float or ndarray
            First transverse direction in meters

        y: float or ndarray
            Second transverse direction in meters

        t_modified: float
            Time in seconds, multiplied by (1-v_antenna/c)
            This multiplication is done in em3dsolver.py, when
            calling the present function.
        """
        # Get the true time
        # (The top.time has been multiplied by (1-v_antenna/c)
        # in em3dsolver.py, before calling the present function)
        t = t_modified/(1.-self.v_antenna/c)
        # Get the position of the antenna at this time
        z_source = self.source_z + self.v_antenna * t

        # When running in the boosted frame, convert these position to
        # the lab frame, so as to use the lab-frame formula of the laser
        if self.boost is not None:
            zlab_source = self.boost.gamma0*( z_source + self.boost.beta0*c*t )
            tlab_source = self.boost.gamma0*( t + self.boost.beta0*z_source/c )
            # Overwrite boosted frame values, within the scope of this function
            z_source = zlab_source
            t = tlab_source

        # Diffraction and stretching factor
        z =  z_source - self.zf
        diffract_factor = 1 - 1j*z*self.inv_zr
        stretch_factor = 1 + \
          4*(self.zeta + self.beta*z)**2 * \
            (self.inv_tau2*self.inv_waist2) / diffract_factor \
        + 2j*(self.phi2 - self.beta**2*self.k0*z) * self.inv_tau2

        # Calculate the argument of the complex exponential
        exp_argument = 1j * self.k0*( c*t -  z_source) \
          - (y**2 + x**2) * self.inv_waist2 / diffract_factor \
          - 1./stretch_factor * self.inv_tau2 * \
            ( t - (z_source - self.z0)/c - self.beta*self.k0*x \
            - 2j*x*(self.zeta + self.beta*z)*self.inv_waist2/diffract_factor )**2

        # Get the profile
        profile = np.exp(exp_argument) / \
          ( diffract_factor**self.geom_coeff * stretch_factor**.5 )

        # Boosted-frame: convert the laser amplitude
        # These formula assume that the antenna is motionless in the lab frame
        if self.boost is not None:
            conversion_factor = 1./self.boost.gamma0
            # The line below is to compensate the fact that the laser
            # amplitude is multiplied by (1-v_antenna/c) in em3dsolver.py
            conversion_factor *= 1./(1. - self.v_antenna/c)
            E0 = conversion_factor * self.E0
        else:
            E0 = self.E0

        return( E0 * profile.real )

class LaguerreGaussianProfile(object):
    """
    Class that calculates a Laguerre-Gaussian laser pulse.
    A typical LG pulse is defined as :

    E(x,y,z) = \left(\frac{r \sqrt{2}}{w} \right)^n L_{mn} \left[
                 \frac{2 r^2}{w^2} \right] e^{- i n \varphi} \; GaussianProfile

    where  r = (x^2 + y^2)^(1/2) and \phi = arctan(y/x).
    n and m are specific parameters to calculate the Laguerre
    polynomial :

    L_{mn} \left[  x \right] = \frac{e^x}{m! \: x^n}  \frac{d^m}{dx^m} \left[
                               e^{-x} x^{n+m} \right]

    Be careful, a new Gouy phase is defined such as :
                \psi_{LG}  = (2m + n+ 1) arctan( xi ) = (2m + n+ 1) \psi_{G}

    Note than when n and m are both equal to 0, this function returns the same
    results as GaussianProfile.
    """

    def __init__( self, m, n, k0, waist, ctau, z0, zf, source_z, source_v, a0,
                  dim, boost, temporal_order = 2 ):

        # Set a number of parameters for the laser
        E0 = a0 * m_e * c**2 * k0/e
        zr = 0.5  * k0 * waist**2
        # Set default focusing position
        if zf is None : zf = z0

        # Store the parameters
        self.m = m
        self.n = n
        self.k0 = k0
        self.waist = waist
        self.zr = zr
        self.ctau = ctau
        self.z0 = z0
        self.E0 = E0
        self.v_antenna = source_v
        self.source_z = source_z
        self.zf = zf
        self.boost = boost
        self.temporal_order = temporal_order

        # Geometric coefficient (for the evolution of the amplitude)
        # In 1D, there is no transverse components, therefore the geomtric
        # coefficient shouldn't be included.
        if  dim=="1d":
            self.geom_coeff = 0.
        elif dim=="2d":
            self.geom_coeff = 0.5
        elif dim in ["circ", "3d"]:
            self.geom_coeff = 1.

    def __call__( self, x, y, t_modified ):
        """
        Return the transverse profile of the laser at the position
        of the antenna

        Parameters:
        -----------
        x: float or ndarray
            First transverse direction in meters

        y: float or ndarray
            Second transverse direction in meters

        t_modified: float
            Time in seconds, multiplied by (1-v_antenna/c)
            This multiplication is done in em3dsolver.py, when
            calling the present function.
        """
        # Calculate the array of radius
        r2 = x**2 + y**2
        r = np.sqrt(r2)

        # phi is the argument of x+i*y
        phi = np.angle(x+1j*y)

        # Get the true time
        # (The top.time has been multiplied by (1-v_antenna/c)
        # in em3dsolver.py, before calling the present function)
        t = t_modified/(1.-self.v_antenna/c)
        # Get the position of the antenna at this time
        z_source = self.v_antenna * t

        # When running in the boosted frame, convert these position to
        # the lab frame, so as to use the lab-frame formula of the laser
        if self.boost is not None:
            zlab_source = self.boost.gamma0*( z_source + self.boost.beta0*c*t )
            tlab_source = self.boost.gamma0*( t + self.boost.beta0*z_source/c )
            # Overwrite boosted frame values, within the scope of this function
            z_source = self.source_z + zlab_source
            t = tlab_source

        # Lab-frame formula for the laser:
        # - Waist and curvature and the position of the source
        z =  z_source - self.zf
        w = self.waist * np.sqrt( 1 + ( z/self.zr )**2 )
        R = z *( 1 + ( self.zr/z )**2 )

        # Generate the Laguerre function via the scipy function
        L_mn = genlaguerre(self.m, self.n)

        # - Propagation phase at the position of the source
        propag_phase = -self.k0 *(c*t) \
             + self.k0 * r2 / (2*R) \
             - self.geom_coeff*(self.n + 2*self.m + 1) *np.arctan( z/self.zr )\
             - phi * self.n

        # - Longitudinal and transverse profile
        trans_profile = np.exp( - r2 / w**2 )
        long_profile = np.exp(
            - (( z_source - c*t + self.z0 ) /self.ctau )**self.temporal_order)
        # -Curvature oscillations
        curvature_oscillations = np.cos( propag_phase )

        # - Prefactor
        prefactor = (self.waist/w)**self.geom_coeff \
                     * (r*np.sqrt(2)/w)**self.n * L_mn(2*(r/w)**2)

        # - Combine profiles
        profile =  prefactor * long_profile * trans_profile \
                    * curvature_oscillations

        # Boosted-frame: convert the laser amplitude
        # These formula assume that the antenna is motionless in the lab frame
        if self.boost is not None:
            conversion_factor = 1./self.boost.gamma0
            # The line below is to compensate the fact that the laser
            # amplitude is multiplied by (1-v_antenna/c) in em3dsolver.py
            conversion_factor *= 1./(1. - self.v_antenna/c)
            E0 = conversion_factor * self.E0
        else:
            E0 = self.E0

        return( E0*profile )
