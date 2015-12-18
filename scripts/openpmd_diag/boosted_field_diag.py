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
- Should we pack the arrays before writing them to the file

- Is it better to write all the attributes with only one proc ?
"""
import os
import numpy as np
from scipy.constants import c
from field_diag import FieldDiagnostic
from field_extraction import get_dataset
from data_dict import z_offset_dict

class BoostedFieldDiagnostic(FieldDiagnostic):
    """
    Class that writes the fields *in the lab frame*, from 
    a simulation in the boosted frame

    Usage
    -----
    After initialization, the diagnostic is called by using the
    `write` method.
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

        # Find the z resolution and size of the diagnostic *in the lab frame*
        # (Needed to initialize metadata in the openPMD file)
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
        # At each timestep, store a slices of the fields in memory buffers 
        self.store_snapshot_slices()

        # Every self.period, write the buffered slices to disk 
        if self.top.it % self.period == 0:
            self.flush_to_disk()
        
    def store_snapshot_slices( self ):
        """
        Store slices of the fields in the memory buffers of the
        corresponding lab snapshots
        """
        # Find the limits of the local subdomain at this iteration
        zmin_boost = self.top.zgrid + self.em.zmminlocal
        zmax_boost = self.top.zgrid + self.em.zmmaxlocal

        # Loop through the labsnapshots
        for snapshot in self.snapshots:

            # Update the positions of the output slice of this snapshot
            # in the lab and boosted frame (current_z_lab and current_z_boost)
            snapshot.update_current_output_positions( self.top.time,
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
                # the results in a properly-formed array
                storing_array = self.slice_handler.extract_slice(
                    self.em, snapshot.current_z_boost, zmin_boost )
                # Register this in the buffers of this snapshot
                snapshot.register_slice( storing_array, self.inv_dz_lab )
        
    def flush_to_disk( self ):
        """
        Writes the buffered slices of fields to the disk

        Erase the buffered slices of the LabSnapshot objects
        """
        # Loop through the labsnapshots and flush the data
        for snapshot in self.snapshots:

            snapshot.buffered_slices = []
            snapshot.buffer_z_indices = []

class LabSnapshot:
    """
    Class that stores data relative to one given snapshot
    in the lab frame (i.e. one given *time* in the lab frame)
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
           Number of the file where this snapshot is to be written
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

    def register_slice( self, field_array, inv_dz_lab ):
        """
        Store the slice of fields represented by field_array
        and also store the z index at which this slice should be
        written in the final lab frame array

        Parameters
        ----------
        field_array: array of reals
            An array of packed fields that corresponds to one slice,
            as given by the SliceHandler object

        inv_dz_lab: float
            Inverse of the grid spacing in z, *in the lab frame*
        """
        # Find the index of the slice in the lab frame
        iz_lab = int( (self.current_z_lab - self.zmin_lab)*inv_dz_lab )

        # Store the values and the index
        self.buffered_slices.append( field_array )
        self.buffer_z_index.append( iz_lab )

    def flush_to_disk( self ):
        
        # Convert slices to array

        # Check that the slices indices are contiguous

        # Gather on the first proc + First proc writes ?
        # Write the data

        # Erase the slices
        self.buffered_slices = []
        self.buffer_z_index = []


class SliceHandler:
    """
    Class that extracts, Lorentz-transforms and writes slices of the fields
    """
    def __init__( self, gamma_boost, beta_boost, dim ):
        """
        Initialize the SliceHandler object

        Parameters
        ----------
        gamma_boost, beta_boost: float
            The Lorentz factor of the boost and the corresponding beta

        dim: string
            Either "2d", "3d", or "circ"
            Indicates the geometry of the fields
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
            
    def extract_slice( self, em, z_boost, zmin_boost ):
        """
        Returns an array that contains the slice of the fields at
        z_boost (the fields returned are already transformed to the lab frame)

        Parameters
        ----------
        em: an EM3DSolver object
            The object from which to extract the fields

        z_boost: float (meters)
            Position of the slice in the boosted frame

        zmin_boost: float (meters)
            Position of the left end of physical part of the local subdomain
            (i.e. excludes guard cells)
        
        Returns
        -------
        An array of reals that packs together the slices of the
        different fields.

        The first index of this array corresponds to the field type
        (10 different field types), and the correspondance
        between the field type and integer index is given self.field_to_index

        The shape of this arrays is:
        - (10, em.nxlocal+1,) for dim="2d"
        - (10, em.nxlocal+1, em.nylocal+1) for dim="3d"
        - (10, 2*em.circ_m+1, em.nxlocal+1) for dim="circ"
        """
        # Extract a slice of the fields *in the boosted frame*
        # at z_boost, using interpolation, and store them in an array
        # (See the docstring of the extract_slice_boosted_frame for
        # the shape of this array.)
        field_array = self.extract_slice_boosted_frame(
            em, z_boost, zmin_boost )

        # Perform the Lorentz transformation of the fields *from
        # the boosted frame to the lab frame*
        self.transform_fields_to_lab_frame( field_array )
            
        return( field_array )

    def extract_slice_boosted_frame( self, em, z_boost, zmin_boost ):
        """
        Extract a slice of the fields at z_boost, using interpolation in z

        See the docstring of extract_slice for the parameters.

        Returns
        -------
        An array that packs together the slices of the different fields.
            The shape of this arrays is:
            - (10, em.nxlocal+1,) for dim="2d"
            - (10, em.nxlocal+1, em.nylocal+1) for dim="3d"
            - (10, 2*em.circ_m+1, em.nxlocal+1) for dim="circ"
        """
        # Allocate an array of the proper shape
        if self.dim=="2d":
            field_array = np.empty( (10, em.nxlocal+1,) )
        elif self.dim=="3d":
            field_array = np.empty( (10, em.nxlocal+1, em.nylocal+1) )
        elif self.dim=="circ":
            field_array = np.empty( (10, 2*em.circ_m+1, em.nxlocal+1) )

        # Find the index of the slice in the boosted frame
        # and the corresponding interpolation shape factor
        dz = em.dz
        # Centered
        z_centered_gridunits = ( z_boost - zmin_boost )/dz
        iz_centered = int( z_centered_gridunits )
        Sz_centered = iz_centered + 1 - z_centered_gridunits
        # Staggered
        z_staggered_gridunits = ( z_boost - zmin_boost - 0.5*dz )/dz
        iz_staggered = int( z_staggered_gridunits )
        Sz_staggered = iz_staggered + 1 - z_staggered_gridunits

        # Shortcut for the correspondance between field and integer index
        f2i = self.field_to_index
        
        # Loop through the fields, and extract the proper slice for each field
        for quantity in self.field_to_index.keys():
            # Here typical values for `quantity` are e.g. 'Er', 'Bx', 'rho'

            # Choose the index and interpolating factor, depending
            # on whether the field is centered in z or staggered
            # - Centered field in z
            if z_offset_dict[quantity] == 0:
                iz = iz_centered
                Sz = Sz_centered
            # - Staggered field in z
            elif z_offset_dict[quantity] == 0.5:
                iz = iz_staggered
                Sz = Sz_staggered
            else:
                raise ValueError( 'Unknown staggered offset for %s: %f' %(
                    quantity, z_offset_dict[quantity] ))

            # Interpolate the centered field in z
            # (Transversally-staggered fields are also interpolated
            # to the nodes of the grid, thanks to the flag transverse_centered)
            field_array[ f2i[quantity], ... ] = Sz * get_dataset(
                self.dim, em, quantity, lgather=False, iz_slice=iz,
                transverse_centered=True )
            field_array[ f2i[quantity], ... ] += (1.-Sz) * get_dataset(
                self.dim, em, quantity, lgather=False, iz_slice=iz+1,
                transverse_centered=True )

        return( field_array )

    def transform_fields_to_lab_frame( self, fields ):
        """
        Modifies the array `fields` in place, to transform the field values
        from the boosted frame to the lab frame.

        The transformation is a transformation with -beta_boost, thus
        the corresponding formulas are:
        - for the transverse part of E and B:
        $\vec{E}_{lab} = \gamma(\vec{E} - c\vec{\beta} \times\vec{B})$
        $\vec{B}_{lab} = \gamma(\vec{B} + \vec{\beta}/c \times\vec{E})$
        - for rho and Jz:
        $\rho_{lab} = \gamma(\rho + \beta J_{z}/c)$
        $J_{z,lab} = \gamma(J_z + c\beta \rho)$
            
        Parameter
        ---------
        fields: array of floats
             An array that packs together the slices of the different fields.
            The shape of this arrays is:
            - (10, em.nxlocal+1,) for dim="2d"
            - (10, em.nxlocal+1, em.nylocal+1) for dim="3d"
            - (10, 2*em.circ_m+1, em.nxlocal+1) for dim="circ"
        """
        # Some shortcuts
        gamma = self.gamma_boost
        cbeta = c*self.beta_boost
        beta_c = self.beta_boost/c
        # Shortcut to give the correspondance between field name
        # (e.g. 'Ex', 'rho') and integer index in the array
        f2i = self.field_to_index
        
        # Lorentz transformations
        # For E and B
        # (NB: Ez and Bz are unchanged by the Lorentz transform)
        if self.dim in ["2d", "3d"]:
            # Use temporary arrays when changing Ex and By in place
            ex_lab = gamma*( fields[f2i['Ex']] + cbeta * fields[f2i['By']] )
            by_lab = gamma*( fields[f2i['By']] + beta_c * fields[f2i['Ex']] ) 
            fields[ f2i['Ex'], ... ] = ex_lab
            fields[ f2i['By'], ... ] = by_lab
            # Use temporary arrays when changing Ey and Bx in place
            ey_lab = gamma*( fields[f2i['Ey']] - cbeta * fields[f2i['Bx']] )
            bx_lab = gamma*( fields[f2i['Bx']] - beta_c * fields[f2i['Ey']] ) 
            fields[ f2i['Ey'], ... ] = ey_lab
            fields[ f2i['Bx'], ... ] = bx_lab
        elif self.dim=="circ":
            # Use temporary arrays when changing Er and Bt in place
            er_lab = gamma*( fields[f2i['Er']] + cbeta * fields[f2i['Bt']] )
            bt_lab = gamma*( fields[f2i['Bt']] + beta_c * fields[f2i['Er']] ) 
            fields[ f2i['Er'], ... ] = er_lab
            fields[ f2i['Bt'], ... ] = bt_lab
            # Use temporary arrays when changing Et and Br in place
            et_lab = gamma*( fields[f2i['Et']] - cbeta * fields[f2i['Br']] )
            br_lab = gamma*( fields[f2i['Br']] - beta_c * fields[f2i['Et']] ) 
            fields[ f2i['Et'], ... ] = et_lab
            fields[ f2i['Br'], ... ] = br_lab
        # For rho and J
        # (NB: the transverse components of J are unchanged)
        # Use temporary arrays when changing rho and Jz in place
        rho_lab = gamma*( fields[f2i['rho']] + beta_c * fields[f2i['Jz']] )
        Jz_lab =  gamma*( fields[f2i['Jz']] + cbeta * fields[f2i['rho']] )
        fields[ f2i['rho'], ... ] = rho_lab
        fields[ f2i['Jz'], ... ] = Jz_lab
