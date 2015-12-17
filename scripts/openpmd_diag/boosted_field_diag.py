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
            self.create_file_empty_meshes( snapshot.filename, i,
                snapshot.t_lab, Nz, snapshot.zmin_lab, dz_lab, self.top.dt )

        # Create a slice handler, which will do all the extraction, Lorentz
        # transformation, etc for each slice to be registered in a
        # LabSnapshot, and abstracts the dimension
        self.slice_handler = SliceHandler(
            self.gamma_boost, self.beta_boost, self.dim )

    def write( self ):
        """
        Redefines the method write of the parent class FieldDiagnostic

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

            # For this snapshot:
            # - check if the output position *in the boosted frame*
            #   is in the current local domain
            # - check if the output position *in the lab frame*
            #   is within the lab-frame boundaries of the current snapshot
            if ( (snapshot.current_z_boost > zmin_boost) and \
                 (snapshot.current_z_boost < zmax_boost) and \
                 (snapshot.current_z_lab > snapshot.zmin_lab) and \
                 (snapshot.current_z_lab < snapshot.zmax_lab) ):

                # In this case, extract the proper slice from the field array,
                # perform a Lorentz transform to the lab frame, and store
                # the results in a proper array
                storing_array = self.slice_handler.extract_slice(
                    self.em, snapshot.current_z_boost, zmin_boost )
                # Register this in the buffers of this snapshot
                snapshot.register_slice( storing_array )
        
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

        # Buffered field slice and corresponding array index in z
        self.buffered_slices = []
        self.buffer_z_index = []

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
        # Find the index of the slice in the lab frame
        i_lab = int( (self.current_z_lab - self.zmin_lab)*self.inv_dz_lab )

    def flush_to_disk( self ):
        
        # Convert slices to array

        # Check that the slices indices are contiguous

        # Gather on the first proc + First proc writes ?
        # Write the data

        # Erase the slices
        self.buffered_slices = []
        self.buffer_z_index = []


class SliceHandler():
    """
    Extracts compacts, transforms and writes the slices
    ### DOC
    """
    def __init__( gamma_boost, beta_boost, dim ):
        """
        ### DOC
        """
        # Store the arguments
        self.dim = dim
        self.gamma_boost = gamma_boost
        self.beta_boost = beta_boost

        # Create a dictionary that contains the correspondance
        # between the field names and array index
        if (dim=="2d") or (dim=="3d") :
            self.field_to_index = {'Ex':0, 'Ey':1, 'Ez':2, 'Bx':3,
                'By':4, 'Bz':5, 'Jx':6, 'Jy':7, 'Jz':8, 'rho':9}
        elif dim=="circ":
            self.field_to_index = {'Er':0, 'Et':1, 'Ez':2, 'Br':3,
                'Bt':4, 'Bz':5, 'Jr':6, 'Jt':7, 'Jz':8, 'rho':9}            

        # Create the reverse dictionary
        self.index_to_field = { value:key for key, value \
                                in self.field_to_index.items() }
            
    def extract_slice( self, em, z_boost ):
        """
        ### DOC ###
        """
        # Find the index of the slice in the boosted frame
        # and the corresponding interpolation shape factor
        # NB: all the fields are node-centered here, because
        # the arrays, Exp, Eyp, Ezp, etc. are used
        iz_boost = int( ( z_boost - zmin_boost )/self.em.dz )
        Sz_boost = iz_boost + 1 - ( z_boost - zmin_boost )/self.em.dz
        # The slice at index iz_boost is weighted with Sz_boost
        # and the slice at index iz_boost+1 is weighted with (1-Sz_boost)

        # Extract the proper arrays
        if (self.dim == "2d") or (self.dim == "3d"):
            storing_array = self.extract_slice_cartesian(i_boost)
        elif self.dim == "circ":
            storing_array = self.extract_slice_circ(i_boost)

        return( storing_array )

    def get_circ_slice( self, quantity, S, iz_slice ):
        """
        Store the slice in an array that contains the different fields,
        in a layout that is close the final layout in the openPMD file

        ### DOC ###
        """
        # Obtain the slices, in the boosted frame
        # In 2D, the slices are 1D arrays (corresponding to the x direction)
        # In 3D, the slices are 2D arrays (x and y directions)
        # (The functions em.getXX simply removes the guard cells)

        # Calculate the S_centered and iz_centered +
        # the S_staggered and iz_staggered (cf fbpic)

        # Interpolate (loop over the fields in an abstract manner)
        interpolate_slice( field, S, iz, centered_in_x, centered_in_y )
        
        # Allocate the boosted_array (form depends on the dimension)

        # Perform the transformation to the lab, in a new array
        boosted_array[ field_to_index[quantity] ] = \
          Sz_boost

        ( em.getex(), iz_slice, S )
        ey_slice = em.getey()[...,iz_slice]
        ez_slice = em.getez()[...,iz_slice]
        bx_slice = em.getbx()[...,iz_slice]
        by_slice = em.getby()[...,iz_slice]
        bz_slice = em.getbz()[...,iz_slice]
        jx_slice = em.getjx()[...,iz_slice]
        jy_slice = em.getjy()[...,iz_slice]
        jz_slice = em.getjz()[...,iz_slice]
        rho_slice = em.getrho()[...,iz_slice]

        # Allocate an array to store them
        if self.dim == "2d":
            boost_array = np.array( (10, ex_slice.shape[0]) )
            lab_array = np.array( (10, ex_slice.shape[0]) )
        if self.dim == "3d":
            lab_array = np.array( (10, ex_slice.shape[0],
                                       ex_slice.shape[1]) )
            boost_array = np.array( (10, ex_slice.shape[0],
                                       ex_slice.shape[1]) )

        # Perform the Lorentz transformation from the boosted frame
        # to the lab frame and store the result at the proper index
        # (correspondance given by field_to_index)
        # Some shortcuts
        gamma = self.gamma_boost
        cbeta = c*self.beta_boost
        beta_c = self.beta_boost/c
        f2i = self.field_to_index
        # Lorentz transformations
        # For E
        lab_array[ f2i['Ez'], ... ] = boost_arrays
        storing_array[ field_to_index['Ex'], ... ] = \
            gamma*( ex_slice + cbeta*by_slice )
        storing_array[ field_to_index['Ey'], ... ] = \
            gamma*( ey_slice - cbeta*bx_slice )
        # For B
        storing_array[ field_to_index['Bz'], ... ] = bz_slice
        storing_array[ field_to_index['Bx'], ... ] = \
            gamma*( bx_slice - beta_c*ey_slice )
        storing_array[ field_to_index['By'], ... ] = \
            gamma*( by_slice + beta_c*ex_slice )
        # For J
        storing_array[ field_to_index['Jz'], ... ] = \
            gamma*( jz_slice + cbeta*rho_slice )
        storing_array[ field_to_index['Jx'], ... ] = jx_slice
        storing_array[ field_to_index['Jy'], ... ] = jy_slice
        # For rho
        storing_array[ field_to_index['rho'], ... ] = \
            gamma*( rho_slice + beta_c*jz_slice )
        
        return( storing_array )

    def extract_slice_cartesian( self, iz_slice ):
        """
        Store the slice in an array that contains the different fields,
        in a layout that is close the final layout in the openPMD file

        ### DOC ###
        """
        # Obtain the slices, in the boosted frame
        # In 2D, the slices are 1D arrays (corresponding to the x direction)
        # In 3D, the slices are 2D arrays (x and y directions)
        # (The functions em.getXX simply removes the guard cells)
        ex_slice = em.getex()[...,iz_slice]
        ey_slice = em.getey()[...,iz_slice]
        ez_slice = em.getez()[...,iz_slice]
        bx_slice = em.getbx()[...,iz_slice]
        by_slice = em.getby()[...,iz_slice]
        bz_slice = em.getbz()[...,iz_slice]
        jx_slice = em.getjx()[...,iz_slice]
        jy_slice = em.getjy()[...,iz_slice]
        jz_slice = em.getjz()[...,iz_slice]
        rho_slice = em.getrho()[...,iz_slice]

        # Allocate an array to store the fields
        # First index: field (correspondance is given by self.field_to_index)
        # Second index: x coordinate
        # Third index (3d only): y coordinate
        if self.dim == "2d":
            storing_array = np.array( (10, ex_slice.shape[0]) )
        if self.dim == "3d":
            storing_array = np.array( (10, ex_slice.shape[0],
                                       ex_slice.shape[1]) )

        # Perform the Lorentz transformation from the boosted frame
        # to the lab frame and store the result at the proper index
        # (correspondance given by field_to_index)
        # Some shortcuts
        gamma = self.gamma_boost
        cbeta = c*self.beta_boost
        beta_c = self.beta_boost/c
        field_to_index = self.field_to_index
        # Lorentz transformations
        # For E
        storing_array[ field_to_index['Ez'], ... ] = ez_slice
        storing_array[ field_to_index['Ex'], ... ] = \
            gamma*( ex_slice + cbeta*by_slice )
        storing_array[ field_to_index['Ey'], ... ] = \
            gamma*( ey_slice - cbeta*bx_slice )
        # For B
        storing_array[ field_to_index['Bz'], ... ] = bz_slice
        storing_array[ field_to_index['Bx'], ... ] = \
            gamma*( bx_slice - beta_c*ey_slice )
        storing_array[ field_to_index['By'], ... ] = \
            gamma*( by_slice + beta_c*ex_slice )
        # For J
        storing_array[ field_to_index['Jz'], ... ] = \
            gamma*( jz_slice + cbeta*rho_slice )
        storing_array[ field_to_index['Jx'], ... ] = jx_slice
        storing_array[ field_to_index['Jy'], ... ] = jy_slice
        # For rho
        storing_array[ field_to_index['rho'], ... ] = \
            gamma*( rho_slice + beta_c*jz_slice )
        
        return( storing_array )
        
    def extract_slice_cylindrical( self, iz_slice ):
        """
        Store the slice in an array that contains the different fields,
        in a layout that is close the final layout in the openPMD file

        ### DOC ###
        """
        # Obtain the slices, in the boosted frame
        # For the mode 0 (1d slice: corresponding to the radial direction)
        er_mode0_slice = em.getex()[:,iz_slice]
        et_mode0_slice = em.getey()[:,iz_slice]
        ez_mode0_slice = em.getez()[:,iz_slice]
        br_mode0_slice = em.getbx()[:,iz_slice]
        bt_mode0_slice = em.getby()[:,iz_slice]
        bz_mode0_slice = em.getbz()[:,iz_slice]
        jr_mode0_slice = em.getjx()[:,iz_slice]
        jt_mode0_slice = em.getjy()[:,iz_slice]
        jz_mode0_slice = em.getjz()[:,iz_slice]
        rho_mode0_slice = em.getrho()[:,iz_slice]
        # For higher modes (2d slice: first index=r, last index=mode)
        er_modes_slice = em.getex_circ()[:,iz_slice,:]
        et_modes_slice = em.getey_circ()[:,iz_slice,:]
        ez_modes_slice = em.getez_circ()[:,iz_slice,:]
        br_modes_slice = em.getbx_circ()[:,iz_slice,:]
        bt_modes_slice = em.getby_circ()[:,iz_slice,:]
        bz_modes_slice = em.getbz_circ()[:,iz_slice,:]
        jr_modes_slice = em.getjx_circ()[:,iz_slice,:]
        jt_modes_slice = em.getjy_circ()[:,iz_slice,:]
        jz_modes_slice = em.getjz_circ()[:,iz_slice,:]
        rho_modes_slice = em.getrho_circ()[:,iz_slice,:]

        # Allocate an array to store the fields in format close
        # to the final openPMD format
        # First index 
        

        # Perform the Lorentz transformation from the boosted frame
        # to the lab frame and store the result at the proper index
        # (correspondance given by field_to_index)
        # Some shortcuts
        gamma = self.gamma_boost
        cbeta = c*self.beta_boost
        beta_c = self.beta_boost/c
        field_to_index = self.field_to_index
        # Lorentz transformations
        # For E
        storing_array[ field_to_index['Ez'], :, :] = ez_slice
        storing_array[ field_to_index['Er'], :, :] = \
            gamma*( er_slice + cbeta*bt_slice )
        storing_array[ field_to_index['Et'], :, :] = \
            gamma*( et_slice - cbeta*br_slice )
        # For B
        storing_array[ field_to_index['Bz'], :, :] = bz_slice
        storing_array[ field_to_index['Bx'], :, :] = \
            gamma*( br_slice - beta_c*et_slice )
        storing_array[ field_to_index['By'], :, :] = \
            gamma*( bt_slice + beta_c*er_slice )
        # For J
        storing_array[ field_to_index['Jz'], :, :] = \
            gamma*( jz_slice + cbeta*rho_slice )
        storing_array[ field_to_index['Jr'], :, :] = jr_slice
        storing_array[ field_to_index['Jt'], :, :] = jt_slice
        # For rho
        storing_array[ field_to_index['rho'], ... ] = \
            gamma*( rho_slice + beta_c*jz_slice )
        
        return( storing_array )
