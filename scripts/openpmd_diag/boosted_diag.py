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
- Which datastructure to use for the slices: dictionary of arrays?
e.g. {'Er':np.array, 'Et':np.array, ...}
- Should one use the IO collectives when only a few proc modify a given file?
- Should we just have the proc writing directly to the file ? Should we gather on the first proc ?
- Should we keep all the files open simultaneously,
or open/close them on the fly?
- What to do for the particles ? How to know the total
number of particles in advance (Needed to create the dataset) ?
"""
import os
import numpy as np

class LabFrameDiagnostic(FieldDiagnostic) :
    """
    ### DOC ###
    """
    def __init__(self, zmin_lab, zmax_lab, v_lab, dt_snapshot_lab,
                 Ntot_snapshot_lab, gamma_boost, period, em, top, w3d,
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

        dt_snapshot_lab: float (seconds)
            Time interval *in the lab frame* between two successive snapshots

        Ntot_snapshots_lab: int
            Total number of snapshots that this diagnostic will produce

        period: int
            Number of iterations for which the data is accumulated in memory,
            before finally writing it to the disk. 
            
        ### Complete this doc ###         
        """
        # Do not leave write_dir as None, as this may conflict with
        # the default directory ('./') in which diagnostics in the
        # boosted frame are written.
        if write_dir is None:
            write_dir='./in_lab_frame'

        # Initialize the normal attributes of a FieldDiagnostic
        FieldDiagnostic.__init__(self, period, em, top, w3d,
                comm_world, fieldtypes, write_dir, lparallel_output)
                
        # Create the list of LabSnapshot objects
        self.snapshots = []
        for i in range( Ntot_snapshots_lab ):
            t_lab = i * dt_snapshots_lab
            snapshot = LabSnapshot( t_lab,
                                    zmin_lab + v_lab*t_lab,
                                    zmax_lab + v_lab*t_lab,
                                    self.write_dir, i )
            self.snapshots.append( snapshot )

        # Determine z resolution and size of the datasets
        self.gamma_boost = gamma_boost
        self.beta_boost = np.sqrt( 1. - 1./gamma_boost**2 )
        self.dz_lab = ### Complete code here
        self.dz_boosted = ### Complete code here
        Nz = int( (zmax - zmin)/self.dz_boosted )
        
        # Create empty openPMD file and datasets
        for snapshot in self.snapshots:
            ### The method below should be defined for FieldDiagnostic
            self.create_empty_openpmd_file( snapshot.filename, Nz )

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
        # Loop through the labsnapshots
        # (for snapshot in snapshots:)
        
        # Check if the z_boosted of this snaphot is in the present local
        # domain, in the boosted frame (using w3d.xmminlocal, etc.)
        # >> If yes, deduce its index in the local domain

        # Check if the z_lab of this snaphot is within the bounds of
        # the moving window (using snapshot.zmin_lab, snapshot.zmax_lab)
        # >> If yes, deduce the index of the present slice in the lab frame

        # If yes to both previous questions, call the
        # corresponding register slice
        # (snapshot.register_slice( i_boosted, i_lab ))

    def flush_to_disk( self ):
        """
        """
        # Loop through the labsnapshots and flush the data
        for snapshot in snapshots:
            snapshot.flush_to_disk()


class LabSnapshot( object ):
    """
    ### DOC ###
    """
    def __init___(self, t_lab, zmin_lab, zmax_lab, write_dir, i):
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
        self.filename = os.path.join( write_dir, 'diags/hdf5/data%05d.png' %i)
        self.zmin_lab = zmin_lab
        self.zmax_lab = zmax_lab
        self.t_lab = t_lab

        self.slices = [ ]
        # Slice should also be an object that contains many fields:
        # Ex, Ey, Ez, rho, etc.
    
        self.slice_indices = [ ]

    def register_slice( self ):

        # Get the right slice for rho, J, E, B

        # Do the conversion to the lab frame

        # Add the slice to the data

    def flush_to_disk( self ):
        
        # Convert slices to array

        # Check that the slices indices are contiguous

        # Gather on the first proc + First proc writes ?
        # Write the data

        # Erase the slices
        self.slice_indices = []
        self.slices = []

        
