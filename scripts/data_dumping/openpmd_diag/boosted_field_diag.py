"""
This file defines the class BoostedFieldDiagnostic

Major features:
- The class reuses the existing methods of FieldDiagnostic
  as much as possible, through class inheritance
- The class implements memory buffering of the slices, so as
  not to write to disk at every timestep

Remaining questions:
- Should one use the IO collectives when only a few proc modify a given file?
- Should we just have the proc writing directly to the file ?
  Should we gather on the first proc ?
- Is it better to write all the attributes of the openPMD file
  with only one proc ?
"""
import os
import numpy as np
import time
from scipy.constants import c
from field_diag import FieldDiagnostic
from field_extraction import get_dataset
from data_dict import z_offset_dict
from parallel import gatherarray

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
        # Record the time it takes
        if self.rank == 0:
            measured_start = time.clock()
            print('\nInitializing the lab-frame diagnostics: %d files...' %(
                Ntot_snapshots_lab) )
        # Loop through the lab snapshots and create the corresponding files
        for i in range( Ntot_snapshots_lab ):
            t_lab = i * dt_snapshots_lab
            snapshot = LabSnapshot( t_lab,
                                    zmin_lab + v_lab*t_lab,
                                    zmax_lab + v_lab*t_lab,
                                    self.write_dir, i, self.rank)
            self.snapshots.append( snapshot )
            # Initialize a corresponding empty file
            if self.lparallel_output == False and self.rank == 0:
                self.create_file_empty_meshes( snapshot.filename, i,
                    snapshot.t_lab, Nz, snapshot.zmin_lab, dz_lab, self.top.dt )
        # Print a message that records the time for initialization
        if self.rank == 0:
            measured_end = time.clock()
            print('Time taken for initialization of the files: %.5f s' %(
                measured_end-measured_start) )

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
                slice_array = self.slice_handler.extract_slice(
                    self.em, snapshot.current_z_boost, zmin_boost )
                # Register this in the buffers of this snapshot
                snapshot.register_slice( slice_array, self.inv_dz_lab )

    def flush_to_disk( self ):
        """
        Writes the buffered slices of fields to the disk

        Erase the buffered slices of the LabSnapshot objects
        """
        # Loop through the labsnapshots and flush the data
        for snapshot in self.snapshots:

            # Compact the successive slices that have been buffered
            # over time into a single array
            field_array, iz_min, iz_max = snapshot.compact_slices()

            # Erase the memory buffers
            snapshot.buffered_slices = []
            snapshot.buffer_z_indices = []

            if self.comm_world is not None:
                # In MPI mode: gather and an array containing the number 
                # of particles on each process
                # Attribute values to iz_min, iz_max and size of the field
                # array if field array is None 
                if field_array is None:
                    iz_min = 0
                    iz_max = 0
                    flat_field_array = np.zeros(0)
                    nx_field_array = 0
                    if self.dim == "3d":
                        ny_field_array = 0

                else:
                    flat_field_array = field_array.flatten()
                    if self.dim in ["2d","3d"]:
                        nx_field_array = np.shape(field_array)[1]
                    elif self.dim == "3d":
                        ny_field_array = np.shape(field_array)[2]
                    elif self.dim == "circ":
                        nx_field_array = np.shape(field_array)[2]
                        
                # Gather the size of the field array
                nx_rank = np.array(
                    self.comm_world.allgather(nx_field_array))
                if self.dim == "3d":
                    ny_rank = np.array(
                        self.comm_world.allgather(ny_field_array))
                
                # Gather arrays, iz_min and iz_max
                g_ar = gatherarray(flat_field_array, root=0, 
                    comm=self.comm_world)
                g_iz_min = np.array(self.comm_world.allgather(iz_min))
                g_iz_max = np.array(self.comm_world.allgather(iz_max))
                
                if self.rank == 0:
                    # Ternary equation: test if field array is None. If not none, 
                    # attribute the global size of Nx, else attribute 0
                    
                    Nx = (self.top.fsdecomp.nxglobal + 1) if g_ar.size!=0 else 0
                    if self.dim == "3d":
                        Ny = (self.top.fsdecomp.nyglobal + 1) \
                        if g_ar.size!=0 else 0
                    elif self.dim == "circ":
                        Ncirc = 2*self.em.circ_m + 1
                     
                    n_slice = 0
                    # Don't have to specify the dimension specifically because
                    # if one of the dimensions does not contain any non null 
                    # value, that implies void
                    if Nx != 0: 
                        iz_min = min([n for n in g_iz_min if n>0]) \
                            if g_iz_min.any() else 0
                        iz_max = max([n for n in g_iz_max if n>0]) \
                            if g_iz_max.any() else 0
                        n_slice = iz_max - iz_min

                    # Create an empty global field array, the one to be written 
                    # in the disk 
                    if self.dim == "2d":
                        f_ar = np.empty((10, Nx, n_slice))
                    elif self.dim == "3d":
                        f_ar = np.empty((10, Nx, Ny, n_slice))
                    elif self.dim == "circ":
                        f_ar = np.empty((10, Ncirc, Ny, n_slice))

                    # indx as index to determine which chunk of field_array
                    # in x-direction comes from i processor
                    indx = 0

                    # indy as index to determine which chunk of field_array
                    # in y-direction comes from i processor
                    if self.dim == "3d":
                        indy = 0
                
                    # sind as index to determine the slice it corresponds 
                    # to in the global field array
                    sind = 0

                    # Loop through all the processors to do the reshaping
                    for i in xrange(self.top.nprocs):
                        s = g_iz_max[i] - g_iz_min[i]

                        if nx_rank[i] !=0 :
                            # gxind as index to determine the slice it 
                            # corresponds to in the x-direction in the global 
                            # field array, valid for 2d, 3d and circ

                            gxind = self.top.fsdecomp.ix[i%self.top.nxprocs]

                            if self.dim =="2d":
                                f_ar[:,gxind:gxind+nx_rank[i],sind:sind+s] = np.reshape(
                                    g_ar[indx:indx+10*nx_rank[i]*s], (10,nx_rank[i],s))
                            elif self.dim =="3d":
                                # gyind: index only valid in 3d 
                                gyind = self.top.fsdecomp.iy[i%self.top.nyprocs]
                                f_ar[:,gxind:gxind+nx_rank[i],gyind:gyind+ny_rank[i] ,sind:sind+s] = np.reshape(
                                    g_ar[indx:indx+10*nx_rank[i]*ny_rank[i]*s], (10,nx_rank[i],ny_rank[i],s))
                                indy += 10*ny_rank[i]*s
                            elif self.dim =="circ":
                                f_ar[:,:,gxind:gxind+nx_rank[i],sind:sind+s] = np.reshape(
                                    g_ar[indx:indx+10*Ncirc*nx_rank[i]*s], (10,Ncirc,nx_rank[i],s))
                            indx += 10*nx_rank[i]*s
                            
                            if (i+1)%self.top.nxprocs==0:
                                sind += s

            else:
                f_ar = field_array

            # Write this array to disk (if this snapshot has new slices)
            if self.rank==0 and f_ar.size!=0:
                self.write_slices( f_ar, iz_min, iz_max,
                    snapshot, self.slice_handler.field_to_index )


    def write_slices( self, field_array, iz_min, iz_max, snapshot, f2i ): 
        """
        For one given snapshot, write the slices of the
        different fields to an openPMD file

        Parameters
        ----------
        field_array: array of reals
            Array of shape
            - (10, em.nxlocal+1, nslices) if dim="2d"
            - (10, em.nxlocal+1, em.nylocal+1, nslices) if dim="3d"
            - (10, 2*em.circ_m+1, em.nxlocal+1, nslices) if dim="circ"

        iz_min, iz_max: integers
            The indices between which the slices will be written
            iz_min is inclusice and iz_max is exclusive

        snapshot: a LabSnaphot object

        f2i: dict
            Dictionary of correspondance between the field names
            and the integer index in the field_array
        """
        # Open the file without parallel I/O in this implementation
        f = self.open_file( snapshot.filename )
        
        field_path = "/data/%d/fields/" %snapshot.iteration
        field_grp = f[field_path]
        
        # Loop over the different quantities that should be written
        for fieldtype in self.fieldtypes:
            # Scalar field
            if fieldtype == "rho":
                data = field_array[ f2i[ "rho" ] ]
                self.write_field_slices( field_grp, data, "rho",
                                         "rho", iz_min, iz_max )
            # Vector field
            elif fieldtype in ["E", "B", "J"]:
                for coord in self.coords:
                    quantity = "%s%s" %(fieldtype, coord)
                    path = "%s/%s" %(fieldtype, coord)
                    data = field_array[ f2i[ quantity ] ]
                    self.write_field_slices( field_grp, data, path,
                                        quantity, iz_min, iz_max )

        # Close the file
        f.close()

    def write_field_slices( self, field_grp, data, path,
                            quantity, iz_min, iz_max ):
        """
        Writes the slices of a given field into the openPMD file

        Parameters
        ----------
        field_grp: an hdf5.Group
            The h5py group that contains all the meshes

        data: array of reals
            An array containing the slices for one given field
            
        path: string
            The path of the dataset to write within field_grp

        quantity: string
            A string that indicates which field is being written
            (e.g. 'Ex', 'Br', or 'rho')

        iz_min, iz_max: integers
            The indices between which the slices will be written
            iz_min is inclusice and iz_max is exclusive
        """
        dset = field_grp[ path ]
        indices = self.global_indices

        # Write the fields depending on the geometry
        if self.lparallel_output:
            if self.dim == "2d":
                dset[ indices[0,0]:indices[1,0], iz_min:iz_max ] = data
            elif self.dim == "3d":
                dset[ indices[0,0]:indices[1,0],
                indices[0,1]:indices[1,1], iz_min:iz_max ] = data
            elif self.dim == "circ":
                # The first index corresponds to the azimuthal mode
                dset[ :, indices[0,0]:indices[1,0], iz_min:iz_max ] = data
           
        else:
            if self.dim == "2d":
                dset[ :, iz_min:iz_max ] = data
            elif self.dim == "3d":
                dset[ :, : , iz_min:iz_max ] = data
            elif self.dim == "circ":
                # The first index corresponds to the azimuthal mode
                dset[ :, :, iz_min:iz_max ] = data

class LabSnapshot:
    """
    Class that stores data relative to one given snapshot
    in the lab frame (i.e. one given *time* in the lab frame)
    """
    def __init__(self, t_lab, zmin_lab, zmax_lab, write_dir, i, rank):
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
        if rank == 0:
            self.filename = os.path.join( write_dir, 'hdf5/data%08d.h5' %i)
        self.iteration = i

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
        self.buffer_z_indices = []

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

    def register_slice( self, slice_array, inv_dz_lab ):
        """
        Store the slice of fields represented by slice_array
        and also store the z index at which this slice should be
        written in the final lab frame array

        Parameters
        ----------
        slice_array: array of reals
            An array of packed fields that corresponds to one slice,
            as given by the SliceHandler object

        inv_dz_lab: float
            Inverse of the grid spacing in z, *in the lab frame*
        """
        # Find the index of the slice in the lab frame
        iz_lab = int( (self.current_z_lab - self.zmin_lab)*inv_dz_lab )

        # Store the values and the index
        self.buffered_slices.append( slice_array )
        self.buffer_z_indices.append( iz_lab )

    def compact_slices(self):
        """
        Compact the successive slices that have been buffered
        over time into a single array, and return the indices
        at which this array should be written.

        Returns
        -------
        field_array: an array of reals of shape
        - (10, em.nxlocal+1, nslices) if dim is "2d"
        - (10, em.nxlocal+1, em.nylocal+1, nslices) if dim is "3d"
        - (10, 2*em.circ_m+1, em.nxlocal+1, nslices) if dim is "circ"
        In the above nslices is the number of buffered slices

        iz_min, iz_max: integers
        The indices between which the slices should be written
        (iz_min is inclusive, iz_max is exclusive)

        Returns None if the slices are empty
        """
        # Return None if the slices are empty
        if len(self.buffer_z_indices) == 0:
            return( None, None, None )
        
        # Check that the indices of the slices are contiguous
        # (This should be a consequence of the transformation implemented
        # in update_current_output_positions, and of the calculation
        # of inv_dz_lab.)
        iz_old = self.buffer_z_indices[0]
        for iz in self.buffer_z_indices[1:]:
            if iz != iz_old - 1:
                raise UserWarning('In the boosted frame diagnostic, '
                        'the buffered slices are not contiguous in z.\n'
                        'The boosted frame diagnostics may be inaccurate.')
                break
            iz_old = iz

        # Pack the different slices together
        # Reverse the order of the slices when stacking the array,
        # since the slices where registered for right to left
        field_array = np.stack( self.buffered_slices[::-1], axis=-1 )

        # Get the first and last index in z
        # (Following Python conventions, iz_min is inclusive,
        # iz_max is exclusive)
        iz_min = self.buffer_z_indices[-1]
        iz_max = self.buffer_z_indices[0] + 1

        return( field_array, iz_min, iz_max )
        
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
        slice_array = self.extract_slice_boosted_frame(
            em, z_boost, zmin_boost )

        # Perform the Lorentz transformation of the fields *from
        # the boosted frame to the lab frame*
        self.transform_fields_to_lab_frame( slice_array )
            
        return( slice_array )

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
            slice_array = np.empty( (10, em.nxlocal+1,) )
        elif self.dim=="3d":
            slice_array = np.empty( (10, em.nxlocal+1, em.nylocal+1) )
        elif self.dim=="circ":
            slice_array = np.empty( (10, 2*em.circ_m+1, em.nxlocal+1) )

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
            slice_array[ f2i[quantity], ... ] = Sz * get_dataset(
                self.dim, em, quantity, lgather=False, iz_slice=iz,
                transverse_centered=True )
            slice_array[ f2i[quantity], ... ] += (1.-Sz) * get_dataset(
                self.dim, em, quantity, lgather=False, iz_slice=iz+1,
                transverse_centered=True )

        return( slice_array )

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
