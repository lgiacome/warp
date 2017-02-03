import numpy as np
from scipy.constants import c, m_e, e
from .boost_tools import BoostConverter
# Import laser antenna and laser profiles
from ..field_solvers.laser import *

def add_laser( em, dim, a0, w0, ctau, z0, zf=None, lambda0=0.8e-6,
               theta_pol=0., source_z=0., zeta=0, beta=0, phi2=0,
               gamma_boost=None, laser_file=None, laser_file_energy=None ):
    """
    Add a linearly-polarized, Gaussian laser pulse in the em object,
    by setting the correct laser_func, laser_emax, laser_source_z
    and laser_polangle

    NB: When using this interface, the antenna is necessarily
    motionless in the lab-frame.

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

    source_z : float (in meters), optional
       The position of the antenna that launches the laser

    zeta: float (in m.s), optional
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       Spatial chirp, at focus,
       as defined in Akturk et al., Opt Express, vol 12, no 19 (2014)

    beta: float (in s), optional
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       Angular dispersion, at focus,
       as defined in Akturk et al., Opt Express, vol 12, no 19 (2014)

    phi2: float (in s^2), optional
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       Temporal chirp, at focus,
       as defined in Akturk et al., Opt Express, vol 12, no 19 (2014)

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
    # i.e. an instance of a class with the __call__ method. This avoids the
    # problem of the EM solver not being picklable if laser_func were an
    # instance method, which is not picklable.

    # - Case of a Gaussian pulse
    if laser_file is None:

        # When running a simulation in boosted frame, convert these parameters
        boost = None
        if (gamma_boost is not None):
            boost = BoostConverter( gamma_boost )
            source_z, = boost.copropag_length([ source_z ],
                                              beta_object=source_v/c)
            source_v, = boost.velocity([ source_v ])

        # Create a laser profile object to store these parameters
        if (beta == 0) and (zeta == 0) and (phi2 == 0):
            # Without spatio-temporal correlations
            laser_profile = GaussianProfile( k0, w0, ctau, z0, zf,
                                source_z, source_v, a0, dim, boost )
        else:
            # With spatio-temporal correlations
            laser_profile = GaussianSTCProfile( k0, w0, ctau, z0, zf,
                source_z, source_v, a0, zeta, beta, phi2, dim, boost )

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
