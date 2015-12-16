"""
This file defines the class FieldDiagnostic
"""
import os
import numpy as np
from generic_diag import OpenPMDDiagnostic

# Import a number of useful dictionaries
from data_dict import circ_dict_quantity, cart_dict_quantity, \
    circ_dict_Jindex, cart_dict_Jindex, \
    field_boundary_dict, particle_boundary_dict, field_solver_dict, \
    x_offset_dict, y_offset_dict, z_offset_dict

class FieldDiagnostic(OpenPMDDiagnostic):
    """
    Class that defines the field diagnostics to be done.

    Usage
    -----
    After initialization, the diagnostic is called by using the
    `write` method.
    """

    def __init__(self, period, em, top, w3d, comm_world=None, 
                 fieldtypes=["rho", "E", "B", "J"], write_dir=None, 
                 lparallel_output=False):
        """
        Initialize the field diagnostic.

        Parameters
        ----------
        period: int
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`)
            
        em: an EM3D object (as defined in em3dsolver)
            Contains the fields data and the different methods to extract it

        top: the object representing the `top` package in Warp
            Contains information on the time.

        w3d: the object representing the `w3d` package in Warp
            Contains the dimensions of the grid.

        comm_world: a communicator object
            Either an mpi4py or a pyMPI object, or None (single-proc)
            
        fieldtypes: a list of strings, optional
            The strings are either "rho", "E", "B" or "J"
            and indicate which field should be written.
            Default: all fields are written
            
        write_dir: string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        lparallel_output: boolean, optional
            Switch to set output mode (parallel or gathering)
            If "True": Parallel output
        """
        # General setup
        OpenPMDDiagnostic.__init__(self, period, top, w3d, comm_world,
                                   lparallel_output, write_dir)

        # Register the arguments
        self.em = em
        self.fieldtypes = fieldtypes

        # Determine the dimensions (Cartesian or cylindrical)
        if self.em.l_2drz is True:
            self.dim = "circ"
        elif self.em.l_2dxz is True:
            self.dim = "2d"
        else:
            self.dim = "3d"
        # Determine the coordinates
        if self.dim == "circ":
            self.coords = ['r', 't', 'z']
        else:
            self.coords = ['x', 'y', 'z']
        
    def write_hdf5( self, iteration ):
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration: int
             The current iteration number of the simulation.
        """
        # Find the file name
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )
        
        # Create the file and setup its attributes
        zmin = self.top.zgrid + self.w3d.zmmin
        self.create_empty_openpmd_file( fullpath, self.top.it,
                self.top.time, self.em.nz, zmin, self.em.dz, self.top.dt )

        # Open the file again, and get the field path
        f = self.open_file( fullpath )
        # (f is None if this processor does not participate is writing data)
        if f is not None:
            field_path = "/data/%d/fields/" %iteration
            field_grp = f[field_path]
        else:
            field_grp = None

        # Loop over the different quantities that should be written
        for fieldtype in self.fieldtypes:
            # Scalar field
            if fieldtype == "rho":
                self.write_dataset( field_grp, "rho", "rho" )
            # Vector field
            elif fieldtype in ["E", "B", "J"]:
                for coord in self.coords:
                    quantity = "%s%s" %(fieldtype, coord)
                    path = "%s/%s" %(fieldtype, coord)
                    self.write_dataset( field_grp, path, quantity )

        # Close the file
        if f is not None:      
            f.close()

    def write_dataset( self, field_grp, path, quantity ):
        """
        Write a given dataset
    
        Parameters
        ----------
        field_grp: an h5py.Group object
            The group that corresponds to the path indicated in meshesPath
        
        path: string
            The relative path where to write the dataset, in field_grp

        quantity: string
            Describes which field is being written.
            (Either rho, Er, Et, Ez, Br, Bz, Bt, Jr, Jt or Jz)
        """
        # Extract the correct dataset
        if field_grp is not None:
            dset = field_grp[path]
        else:
            dset = None
        
        # Circ case
        if self.dim == "circ":
            self.write_circ_dataset( dset, quantity )
        # 2D Cartesian case
        elif self.dim == "2d":
            self.write_cart2d_dataset( dset, quantity )
        # 3D Cartesian case
        elif self.dim == "3d":
            self.write_cart3d_dataset( dset, quantity )

    def write_circ_dataset( self, dset, quantity ):
        """
        Write a dataset in Circ coordinates
        """
        # Fill the dataset with these quantities
        # Gathering mode
        if self.lparallel_output == False:
            F, F_circ, _ = self.get_circ_dataset( quantity, lgather=True )
            if self.rank == 0:
	            # Mode m=0
    	        dset[0,:,:] = F
                # Higher modes (real and imaginary part)
                for m in range(self.em.circ_m):
            	    dset[2*m+1,:,:] = F_circ[:,:,m].real
            	    dset[2*m+2,:,:] = F_circ[:,:,m].imag
        # Parallel mode
        else:
            F, F_circ, bounds = self.get_circ_dataset( quantity, False )
            # Mode m=0
            dset[ 0, bounds[0,0]:bounds[1,0],
                     bounds[0,1]:bounds[1,1] ] = F
            # Higher modes (real and imaginary part)
            for m in range(self.em.circ_m):
                dset[ 2*m+1, bounds[0,0]:bounds[1,0],
                         bounds[0,1]:bounds[1,1] ] = F_circ[:,:,m].real
                dset[ 2*m+2, bounds[0,0]:bounds[1,0],
                         bounds[0,1]:bounds[1,1] ] = F_circ[:,:,m].imag

    def write_cart2d_dataset( self, dset, quantity ):
        """
        Write a dataset in Cartesian coordinates
        """            
        # Fill the dataset with these quantities
        # Gathering mode
        if self.lparallel_output == False:
            F, _ = self.get_cart_dataset( quantity, True )
            if self.rank == 0:
    	        dset[:,:] = F
        # Parallel mode
        else:
            F, bounds = self.get_cart_dataset( quantity, False )
            dset[ bounds[0,0]:bounds[1,0],
                    bounds[0,1]:bounds[1,1] ] = F

    def write_cart3d_dataset( self, dset, quantity ):
        """
        Write a dataset in Cartesian coordinates
        """            
        # Fill the dataset with these quantities
        # Gathering mode
        if self.lparallel_output == False:
            F, _ = self.get_cart_dataset( quantity, True )
            if self.rank == 0:
    	        dset[:,:,:] = F
        # Parallel mode
        else:
            F, bounds = self.get_cart_dataset( quantity, False )
            dset[ bounds[0,0]:bounds[1,0],
                  bounds[0,1]:bounds[1,1],
                  bounds[0,2]:bounds[1,2] ] = F
                     
    def get_circ_dataset( self, quantity, lgather):
        """
        Get a given quantity in Circ coordinates

        Parameters
        ----------
        quantity: string
            Describes which field is being written.
            (Either rho, Er, Et, Ez, Br, Bz, Bt, Jr, Jt or Jz)

        lgather: boolean
            Defines if data is gathered on me (process) = 0
            If "False": No gathering is done
        """
        F_circ = None
        em = self.em
        
        # Treat the currents in a special way
        if quantity in ['Jr', 'Jt', 'Jz']:
            # Get the array index that corresponds to that component
            i = circ_dict_Jindex[ quantity ]
            # Extract mode 0
            F = em.getarray( em.fields.J[:,:,:,i] )
            # Extract higher modes
            if em.circ_m > 0:
                F_circ = em.getarray_circ( em.fields.J_circ[:,:,i,:] )
                
        # Treat the fields E, B, rho in a more systematic way
        elif quantity in ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz', 'rho' ]:
            # Get the field name in Warp
            field_name = circ_dict_quantity[ quantity ]
            # Extract mode 0
            field_array = getattr( em.fields, field_name )
            F = em.getarray( field_array )
            # Extract higher modes
            if em.circ_m > 0:
                field_array = getattr( em.fields, field_name + '_circ')
                F_circ = em.getarray_circ( field_array )

        # Gather array if lgather = True 
        # (Mutli-proc operations using gather)
        # Only done in non-parallel case
        if lgather is True:
            F = em.gatherarray( F )
            if em.circ_m > 0:
                F_circ = em.gatherarray( F_circ )
        
        # Get global positions (indices) of local domain
        # Only needed for parallel output
        if lgather == False:
            nx, nz = np.shape(F)
            bounds = np.zeros([2,2], dtype = np.int)
            bounds[0,0] = int(round((em.block.xmin - em.xmmin) / em.dx))
            bounds[1,0] = bounds[0,0] + nx
            bounds[0,1] = int(round((em.block.zmin - em.zmmin) / em.dz))
            bounds[1,1] = bounds[0,1] + nz
        else:
            bounds = None

        return( F, F_circ, bounds )

    def get_cart_dataset( self, quantity, lgather):
        """
        Get a given quantity in Cartesian coordinates

        Parameters
        ----------
        quantity: string
            Describes which field is being written.
            (Either rho, Er, Et, Ez, Br, Bz, Bt, Jr, Jt or Jz)

        lgather: boolean
            Defines if data is gathered on me (process) = 0
            If "False": No gathering is done
        """
        em = self.em
        
        # Treat the currents in a special way
        if quantity in ['Jx', 'Jy', 'Jz']:
            # Get the array index that corresponds to that component
            i = cart_dict_Jindex[ quantity ]
            F = em.getarray( em.fields.J[:,:,:,i] )

        # Treat the fields E, B, rho in a more systematic way
        elif quantity in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho' ]:
            # Get the field name in Warp
            field_name = cart_dict_quantity[ quantity ]
            field_array = getattr( em.fields, field_name )
            F = em.getarray( field_array )

        # Gather array if lgather = True 
        # (Mutli-proc operations using gather)
        # Only done in non-parallel case
        if lgather is True:
            F = em.gatherarray( F )
        
        # Get global positions (indices) of local domain
        # Only needed for parallel output
        if lgather == False:
            if F.ndim == 2:
                nx, nz = np.shape(F)
                bounds = np.zeros([2,2], dtype = np.int)
                bounds[0,0] = int(round((em.block.xmin - em.xmmin) / em.dx))
                bounds[1,0] = bounds[0,0] + nx
                bounds[0,1] = int(round((em.block.zmin - em.zmmin) / em.dz))
                bounds[1,1] = bounds[0,1] + nz
            elif F.ndim == 3:
                nx, ny, nz = np.shape(F)
                bounds = np.zeros([2,3], dtype = np.int)
                bounds[0,0] = int(round((em.block.xmin - em.xmmin) / em.dx))
                bounds[1,0] = bounds[0,0] + nx
                bounds[0,1] = int(round((em.block.ymin - em.ymmin) / em.dy))
                bounds[1,1] = bounds[0,1] + ny
                bounds[0,2] = int(round((em.block.zmin - em.zmmin) / em.dz))
                bounds[1,2] = bounds[0,2] + nz
        else:
            bounds = None

        return( F, bounds )


    # OpenPMD setup methods
    # ---------------------

    def create_empty_openpmd_file( self, fullpath, iteration,
                                   time, Nz, zmin, dz, dt ):
        """
        Create an openPMD file and setup all its attributes

        Parameters
        ----------
        fullpath: string
            The absolute path to the file to be created

        iteration: int
            The iteration number of this diagnostic

        time: float (seconds)
            The physical time at this iteration

        Nz: int
            The number of gridpoints along z in this diagnostics

        zmin: float (meters)
            The position of the left end of the box
            
        dz: float (meters)
            The resolution in z of this diagnostic

        dt: float (seconds)
            The timestep of the simulation
        """
        # Determine the shape of the datasets that will be written
        # Circ case
        if self.dim == "circ":
            data_shape = ( 2*self.em.circ_m+1, self.em.nx+1, Nz+1 )
        # 2D case
        elif self.dim == "2d":
            data_shape = ( self.em.nx+1, Nz+1 )
        # 3D case
        elif self.dim == "3d":
            data_shape = ( self.em.nx+1, self.em.ny+1, Nz+1 )

        # Create the file
        f = self.open_file( fullpath )
            
        # Setup the different layers of the openPMD file
        # (f is None if this processor does not participate is writing data)
        if f is not None:

            # Setup the attributes of the top level of the file
            self.setup_openpmd_file( f, iteration, time, dt )

            # Setup the meshes group (contains all the fields)
            field_path = "/data/%d/fields/" %iteration
            field_grp = f.require_group(field_path)
            self.setup_openpmd_meshes_group(field_grp)

            # Loop over the different quantities that should be written
            # and setup the corresponding datasets
            for fieldtype in self.fieldtypes:

                # Scalar field
                if fieldtype == "rho":
                    # Setup the dataset
                    dset = field_grp.require_dataset(
                        "rho", data_shape, dtype='f')
                    self.setup_openpmd_mesh_component( dset, "rho" )
                    # Setup the record to which it belongs
                    self.setup_openpmd_mesh_record( dset, "rho", dz, zmin )

                # Vector field
                elif fieldtype in ["E", "B", "J"]:
                    # Setup the datasets
                    for coord in self.coords:
                        quantity = "%s%s" %(fieldtype, coord)
                        path = "%s/%s" %(fieldtype, coord)
                        dset = field_grp.require_dataset(
                            path, data_shape, dtype='f')
                        self.setup_openpmd_mesh_component( dset, quantity )
                    # Setup the record to which they belong
                    self.setup_openpmd_mesh_record( 
                        field_grp[fieldtype], fieldtype, dz, zmin )

                # Unknown field
                else:
                    raise ValueError(
                        "Invalid string in fieldtypes: %s" %fieldtype)

            # Close the file
            f.close()

    def setup_openpmd_meshes_group( self, dset ):
        """
        Set the attributes that are specific to the mesh path
        
        Parameter
        ---------
        dset: an h5py.Group object that contains all the mesh quantities
        """
        # Field Solver
        dset.attrs["fieldSolver"] = field_solver_dict[ self.em.stencil ]
        # Field and particle boundary
        # - 2D and Circ
        if self.em.l_2dxz:
            dset.attrs["fieldBoundary"] = np.array([
                field_boundary_dict[ self.w3d.boundxy ],
                field_boundary_dict[ self.w3d.boundxy ],
                field_boundary_dict[ self.w3d.bound0 ],
                field_boundary_dict[ self.w3d.boundnz ] ])
            dset.attrs["particleBoundary"] = np.array([
                particle_boundary_dict[ self.top.pboundxy ],
                particle_boundary_dict[ self.top.pboundxy ],
                particle_boundary_dict[ self.top.pbound0 ],
                particle_boundary_dict[ self.top.pboundnz ] ])
        # - 3D
        else:
            dset.attrs["fieldBoundary"] = np.array([
                field_boundary_dict[ self.w3d.boundxy ],
                field_boundary_dict[ self.w3d.boundxy ],
                field_boundary_dict[ self.w3d.boundxy ],
                field_boundary_dict[ self.w3d.boundxy ],
                field_boundary_dict[ self.w3d.bound0 ],
                field_boundary_dict[ self.w3d.boundnz ] ])
            dset.attrs["particleBoundary"] = np.array([
                particle_boundary_dict[ self.top.pboundxy ],
                particle_boundary_dict[ self.top.pboundxy ],
                particle_boundary_dict[ self.top.pboundxy ],
                particle_boundary_dict[ self.top.pboundxy ],
                particle_boundary_dict[ self.top.pbound0 ],
                particle_boundary_dict[ self.top.pboundnz ] ])

        # Current Smoothing
        if np.all( self.em.npass_smooth == 0 ):
            dset.attrs["currentSmoothing"] = np.string_("none")
        else:
            dset.attrs["currentSmoothing"] = np.string_("Binomial")
            dset.attrs["currentSmoothingParameters"] = str(self.em.npass_smooth)
        # Charge correction
        dset.attrs["chargeCorrection"] = np.string_("none")


    def setup_openpmd_mesh_record( self, dset, quantity, dz, zmin ):
        """
        Sets the attributes that are specific to a mesh record
        
        Parameter
        ---------
        dset: an h5py.Dataset or h5py.Group object

        quantity: string
            The name of the record (e.g. "rho", "J", "E" or "B")

        dz: float (meters)
            The resolution in z of this diagnostic

        zmin: float (meters)
            The position of the left end of the grid
        """
        # Generic record attributes
        self.setup_openpmd_record( dset, quantity )
        
        # Geometry parameters
        # - thetaMode
        if self.dim == "circ":
            dset.attrs['geometry']  = np.string_("thetaMode")
            dset.attrs['geometryParameters'] = \
              np.string_("m=%d;imag=+" %(self.em.circ_m + 1))
            dset.attrs['gridSpacing'] = np.array([ self.em.dx, dz ])
            dset.attrs['axisLabels'] = np.array([ 'r', 'z' ])
            dset.attrs["gridGlobalOffset"] = np.array([self.w3d.xmmin, zmin])
        # - 2D Cartesian
        elif self.dim == "2d":
            dset.attrs['geometry'] = np.string_("cartesian")
            dset.attrs['gridSpacing'] = np.array([ self.em.dx, dz ])
            dset.attrs['axisLabels'] = np.array([ 'x', 'z' ])
            dset.attrs["gridGlobalOffset"] = np.array([self.w3d.xmmin, zmin])
        # - 3D Cartesian
        elif self.dim == "3d":
            dset.attrs['geometry'] = np.string_("cartesian")
            dset.attrs['gridSpacing'] = np.array([ self.em.dx, self.em.dy, dz ])
            dset.attrs['axisLabels'] = np.array([ 'x', 'y', 'z' ])
            dset.attrs["gridGlobalOffset"] = np.array([ self.w3d.xmmin,
                                                    self.w3d.ymmin, zmin])
        # Generic attributes
        dset.attrs["dataOrder"] = np.string_("C")
        dset.attrs["gridUnitSI"] = 1.
        dset.attrs["fieldSmoothing"] = np.string_("none")

    def setup_openpmd_mesh_component( self, dset, quantity ):
        """
        Set up the attributes of a mesh component
    
        Parameter
        ---------
        dset: an h5py.Dataset
        
        quantity: string
            The field that is being written
        """
        # Generic setup of the component
        self.setup_openpmd_component( dset )
        
        # Field positions
        if (self.em.l_2dxz==True):
            positions = np.array([0., 0.])
        else:
            positions = np.array([0.,0.,0.])
        # Along x
        positions[0] = x_offset_dict[ quantity ]
        # Along y (3D Cartesian only)
        if (self.em.l_2dxz==False):
            positions[1] = y_offset_dict[quantity]
        # Along z
        positions[-1] = z_offset_dict[ quantity ]

        dset.attrs['position'] = positions
