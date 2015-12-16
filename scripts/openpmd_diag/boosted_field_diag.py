"""
Major ideas of this implementation:

- Try to reuse the existing structures (FieldDiagnostic, etc.) as
  much as possible, so that we don't have to rewrite the openPMD attribute

- Implement storing of the data in memory (done at each iteration),
  and flushing to the disk every 10/20 timestep
  (will be very useful on the GPU, to avoid waisting time in CPU communication)

- Encapsulate the data of each lab snapshot (i.e. time in the lab frame,
  file to which it writes, accumulated data in memory) into a dedicated object
  (LabSnapshot)

Questions:
- Which datastructure to use for the slices => array of all fields
- Should one use the IO collectives when only a few proc modify a given file?
- Should we just have the proc writing directly to the file ? Should we gather on the first proc ?
- Should we keep all the files open simultaneously,
or open/close them on the fly?
- What to do for the particles ? How to know the total
number of particles in advance (Needed to create the dataset) ?
>> Do resizable particle arrays

- Can we zero out the fields when we create a new dataset ?
- Is it better to write all the attributes with only one proc ?

"""
import os
import numpy as np
from scipy.constants import c
from field_diag import FieldDiagnostic

class BoostedFieldDiagnostic(FieldDiagnostic):
    """
    ### DOC ###
    """
    def __init__(self, zmin_lab, zmax_lab, v_lab, dt_snapshots_lab,
                 Ntot_snapshots_lab, gamma_boost, period, em, top, w3d,
                 comm_world=None, fieldtypes=["rho", "E", "B", "J"],
                 write_dir=None, lparallel_output=False ) :
        """
        Initialize diagnostics that retrieve the data in the lab frame,
        as a series of snapshot (one file per snapshot),
        within a virtual moving window defined by zmin_lab, zmax_lab, v_lab.
                 
        Parameters
        ----------
        zmin_lab, zmax_lab: floats (meters)
            Positions of the minimum and maximum of the virtual moving window,
            *in the lab frame*, at t=0

        v_lab: float (m.s^-1)
            Speed of the moving window *in the lab frame*

        dt_snapshots_lab: float (seconds)
            Time interval *in the lab frame* between two successive snapshots

        Ntot_snapshots_lab: int
            Total number of snapshots that this diagnostic will produce

        period: int
            Number of iterations for which the data is accumulated in memory,
            before finally writing it to the disk. 
            
        See the documentation of FieldDiagnostic for the other parameters       
        """
        # Do not leave write_dir as None, as this may conflict with
        # the default directory ('./diags') in which diagnostics in the
        # boosted frame are written.
        if write_dir is None:
            write_dir='lab_diags'

        # Initialize the normal attributes of a FieldDiagnostic
        FieldDiagnostic.__init__(self, period, em, top, w3d,
                comm_world, fieldtypes, write_dir, lparallel_output)

        # Register the boost quantities
        self.gamma_boost = gamma_boost
        self.inv_gamma_boost = 1./gamma_boost
        self.beta_boost = np.sqrt( 1. - self.inv_gamma_boost**2 )
        self.inv_beta_boost = 1./self.beta_boost

        # Find the z resolution and size of the diagnostic in the lab frame
        dz_lab = c*self.top.dt * self.inv_beta_boost*self.inv_gamma_boost
        Nz = int( (zmax_lab - zmin_lab)/dz_lab )
        self.inv_dz_lab = 1./dz_lab
        
        # Create the list of LabSnapshot objects
        self.snapshots = []
        for i in range( Ntot_snapshots_lab ):
            t_lab = i * dt_snapshots_lab
            snapshot = LabSnapshot( t_lab,
                                    zmin_lab + v_lab*t_lab,
                                    zmax_lab + v_lab*t_lab,
                                    self.write_dir, i )
            self.snapshots.append( snapshot )
            # Initialize a corresponding empty file
            self.create_empty_openpmd_file( snapshot.filename, i,
                snapshot.t_lab, Nz, snapshot.zmin_lab, dz_lab, self.top.dt )

    def write( self ):
        """
        Redefines the method write of the parent FieldDiagnostic class

        Should be registered with installafterstep in Warp
        """
        # Store snapshots slices in memory at each timestep
        self.store_snapshot_slices()

        # Write the stored slices to disk every self.period
        if self.top.it % self.period == 0:
            self.flush_to_disk()
        
    def store_snapshot_slices( self ):
        """
        """
        # Find the limits of the local subdomain at this iteration
        zmin_boost = self.top.zgrid + self.w3d.zmmin_local
        zmax_boost = self.top.zgrid + self.w3d.zmmax_local

        # Loop through the labsnapshots
        for snapshot in self.snapshots:

            # Update the positions of the output slice of this snapshot
            # in the lab and boosted frame (current_z_lab and current_z_boost)
            snapshot.update_current_output_positions( self.top.t,
                            self.inv_gamma_boost, self.inv_beta_boost )

            # Check if the output position *in the boosted frame*
            # is in the current local domain
            # Check if the output position *in the lab frame*
            # is within the lab-frame boundaries of the current snapshot
            if ( (snapshot.current_z_boost > zmin_boost) and \
                 (snapshot.current_z_boost < zmax_boost) and \
                 (snapshot.current_z_lab > snapshot.zmin_lab) and \
                 (snapshot.current_z_lab < snapshot.zmax_lab) ):

                # In this case register the slice into the memory buffers
                self.register_slice_into( snapshot )

    def register_slice_into( self, snapshot ):
        """
        """
        # Find the index of the slice in the boosted frame
        ### Should be more exact: should take into account the
        ### staggering, the guard cells, and the interpolation
        zmin_boost = self.top.zgrid + self.w3d.zmmin_local
        i_boost = int( ( snapshot.current_z_boost - zmin_boost )/self.em.dz )

        # Find the index of the slice in the lab frame
        i_lab = int( (snaphot.current_z_lab - zmin.zmin_lab)*self.inv_dz_lab )

        # Extract the proper arrays
        if self.dim == "2d":
            array = self.extract_slice_2d(i_boost)
        elif self.dim == "circ":
            array = self.extract_slice_circ(i_boost)
        elif self.dim == "3d":
            array = self.extract_slice_3d(i_boost)

    def extract_array_2d( self, i_slice ):
        """
        """
        

        
            
    def flush_to_disk( self ):
        """
        """
        # Loop through the labsnapshots and flush the data
        for snapshot in self.snapshots:
            snapshot.flush_to_disk()

            
class LabSnapshot( object ):
    """
    ### DOC ###
    """
    def __init__(self, t_lab, zmin_lab, zmax_lab, write_dir, i):
        """
        Initialize a LabSnapshot 

        Parameters
        ----------
        t_lab: float (seconds)
            Time of this snapshot *in the lab frame*
            
        zmin_lab, zmax_lab: floats
            Longitudinal limits of this snapshot

        write_dir: string
            Absolute path to the directory where the data for
            this snapshot is to be written

        i: int
           Number of the file, where this snapshot is to be written
        """
        # Deduce the name of the filename where this snapshot writes
        self.filename = os.path.join( write_dir, 'hdf5/data%05d.h5' %i)

        # Time and boundaries in the lab frame (constants quantities)
        self.zmin_lab = zmin_lab
        self.zmax_lab = zmax_lab
        self.t_lab = t_lab

        # Positions where the fields are to be registered
        # (Change at every iteration)
        self.current_z_lab = 0
        self.current_z_boost = 0

        # Slice should also be an object that contains many fields:
        # Ex, Ey, Ez, rho, etc.
        self.slices = [ ]
        self.slice_indices = [ ]

    def update_current_output_positions( self, t_boost, inv_gamma, inv_beta ):
        """
        Update the positions of output for this snapshot, so that
        if corresponds to the time t_boost in the boosted frame

        Parameters
        ----------
        t_boost: float (seconds)
            Time of the current iteration, in the boosted frame

        inv_gamma, inv_beta: floats
            Inverse of the Lorentz factor of the boost, and inverse
            of the corresponding beta.
        """
        t_lab = self.t_lab

        # This implements the Lorentz transformation formulas,
        # for a snapshot having a fixed t_lab
        self.current_z_boost = ( t_lab*inv_gamma - t_boost )*c*inv_beta
        self.current_z_lab = ( t_lab - t_boost*inv_gamma )*c*inv_beta

    def register_slice( self, em, zmin_boost, inv_dz_lab ):
        """
        """

    def flush_to_disk( self ):
        
        # Convert slices to array

        # Check that the slices indices are contiguous

        # Gather on the first proc + First proc writes ?
        # Write the data

        # Erase the slices
        self.slice_indices = []
        self.slices = []

        
