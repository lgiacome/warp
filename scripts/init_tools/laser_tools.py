import numpy as np
from scipy.constants import c, m_e, e
from scipy.interpolate import RegularGridInterpolator
from .boost_tools import BoostConverter
import h5py
# Try importing parallel functions, in order to broadcast
# the experimental laser file, if required
try:
    from warp.parallel import mpibcast, me
except ImportError:
    # Single-proc simulation
    mpibcast = lambda x:x
    me = 0

def add_laser( em, dim, a0, w0, ctau, z0, zf=None, lambda0=0.8e-6,
               theta_pol=0., source_z=0., gamma_boost=None,
               laser_file=None, laser_file_energy=None ) :
    """
    Add a linearly-polarized, Gaussian laser pulse in the em object,
    by setting the correct laser_func, laser_emax, laser_source_z
    and laser_polangle
    
    Parameters
    ----------
    em : an EM3D object
       The structure that contains the fields of the simulation

    dim: str
       Either "2d", "3d" or "circ"
       
    a0 : float (unitless)
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The a0 of a Gaussian pulse at focus
    
    w0 : float (in meters)
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The waist of the Gaussian pulse at focus

    ctau : float (in meters)
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The "longitudinal waist" (or pulse length) of the Gaussian pulse

    z0 : float (in meters)
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The position of the laser centroid relative to z=0.

    zf : float (in meters), optional
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The position of the laser focus relative to z=0.
       If not provided, then the laser focus is at z0
    
    lambda0 : float (in meters), optional
       The central wavelength of the laser
       Default : 0.8 microns (Ti:Sapph laser)

    theta_pol : float (in radians), optional
       The angle of polarization with respect to the x axis
       Default : 0 rad

    source_z : float, optional
       The position of the antenna that launches the laser
 
    gamma_boost : float, optional
        When initializing the laser in a boosted frame, set the value of
        `gamma_boost` to the corresponding Lorentz factor. All the other
        quantities (ctau, zf, source_z, etc.) are to be given in the lab frame.

    laser_file: str or None
       If None, the laser will be initialized as Gaussian
       Otherwise, the laser_file should point to a standardized HDF5 file
       which contains the following datasets:
       - 't' and 'r': 1D datasets of coordinates, in SI
       - 'Ereal' and 'Eimag': 2D datasets in SI, so that the laser energy is 1J

    laser_file_energy: float or None
       *Used only if a laser_file is provided*
       Total energy of the pulse, in Joules
    """
    # Wavevector and speed of the antenna
    k0 = 2*np.pi/lambda0
    source_v = 0.

    # Create a laser_profile object
    # Note that the laser_profile needs to be a callable instance of a class,
    # i.e. an instance of a class with the __call__ method. This avoids the problem of the
    # EM solver not being picklable if laser_func were an instance method, which is not picklable.

    # - Case of a Gaussian pulse
    if laser_file is None:

        # When running a simulation in boosted frame, convert these parameters
        boost = None
        if (gamma_boost is not None):
            boost = BoostConverter( gamma_boost )
            source_v, = boost.velocity([ 0. ])
            source_z, = boost.static_length([ source_z ])
        
        # Create a laser profile object to store these parameters
        laser_profile = GaussianProfile( k0, w0, ctau, z0, zf,
                                  source_z, a0, dim, boost )

    # - Case of an experimental profile
    else:

        # Reject boosted frame
        if (gamma_boost is not None) and (gamma_boost != 1.):
            raise ValueError('Boosted frame not implemented for '
                             'arbitrary laser profile.')

        # Create a laser profile object
        laser_profile = ExperimentalProfile( k0, laser_file,
                                             laser_file_energy )
    
    # Link its profile function the em object
    em.laser_func = laser_profile

    # Link the rest of the parameters to the em objects
    em.laser_emax = laser_profile.E0
    em.laser_source_z = source_z
    em.laser_source_v = source_v
    em.laser_polangle = theta_pol

    # Additional parameters for the order of deposition of the laser
    em.laser_depos_order_x=1
    em.laser_depos_order_y=1
    em.laser_depos_order_z=1


class ExperimentalProfile( object ):
    """Class that calculates the laser from a data file."""

    def __init__( self, k0, laser_file, laser_file_energy ):
        
        # The first processor loads the file and sends it to the others
        # (This prevents all the processors from accessing the same file,
        # which can substantially slow down the simulation)
        if me==0:
            with h5py.File(laser_file) as f:
                r = f['r'][:]
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

    def __init__( self, k0, w0, ctau, z0, zf, source_z, a0, dim, boost ):

        # Set a number of parameters for the laser      
        E0 = a0*m_e*c**2*k0/e
        zr = 0.5*k0*w0**2
        # Set default focusing position
        if zf is None : zf = z0

        # If there is a boost, change the parameters
        # (NB: source_z is given as already transformed by the boosted frame)
        if boost is not None:
            zr, zf = boost.static_length([ zr, zf ])
            ctau, z0 = boost.copropag_length([ ctau, z0 ])
            k0, E0 = boost.wavenumber([ k0, E0 ])
        
        # Store the parameters
        self.k0 = k0
        self.w0 = w0
        self.zr = zr
        self.ctau = ctau
        self.zf = zf
        self.z0 = z0
        self.source_z = source_z
        self.E0 = E0
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
        r2 = x**2 + y**2
        
        # Waist and curvature and the position of the source
        z = self.source_z - self.zf
        w = self.w0 * np.sqrt( 1 + ( z/self.zr )**2 )
        R = z *( 1 + ( self.zr/z )**2 )

        # When runnning a simulation in boosted frame, convert the curvature
        if self.boost is not None:
            R, = self.boost.curvature([ R ])

        # Propagation phase at the position of the source
        # Even when the source is moving (e.g. boosted frame), self.source_z
        # is the initial position of the source; the fact that the source is
        # moving is taken into account in em3dsolver.py, by multiplying
        # top.time by (1-laser_source_v/c) before calling this function
        propag_phase = self.k0*( self.source_z - c*t) \
            - self.geom_coeff * np.arctan( z/self.zr ) \
            + self.k0 * r2 / (2*R)

        # Longitudinal and transverse profile
        trans_profile = (self.w0/w)**self.geom_coeff * np.exp( - r2 / w**2 )
        long_profile = np.exp(
            - ( self.source_z - c*t - self.z0 )**2 /self.ctau**2 )
        # Curvature oscillations
        curvature_oscillations = np.cos( propag_phase )
        # Combine profiles
        profile =  long_profile * trans_profile * curvature_oscillations
        
        return( self.E0*profile )

        
