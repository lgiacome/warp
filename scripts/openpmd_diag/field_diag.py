"""
This file defines the class FieldDiagnostic
"""
import os
import h5py
import numpy as np
from scipy import constants
from generic_diag import OpenPMDDiagnostic
from parallel import gatherarray

# Correspondance between the names in OpenPMD and the names in Warp
circ_dict_quantity = { 'rho':'Rho', 'Er':'Exp', 'Et':'Eyp', 'Ez':'Ezp', 
                        'Br':'Bxp', 'Bt':'Byp', 'Bz':'Bzp' }
cart_dict_quantity = { 'rho':'Rho', 'Ex':'Exp', 'Ey':'Eyp', 'Ez':'Ezp', 
                        'Bx':'Bxp', 'By':'Byp', 'Bz':'Bzp' }
circ_dict_Jindex = { 'Jr':0, 'Jt':1, 'Jz':2 }
cart_dict_Jindex = { 'Jx':0, 'Jy':1, 'Jz':2 }
    

class FieldDiagnostic(OpenPMDDiagnostic) :
    """
    Class that defines the field diagnostics to be done.

    Usage
    -----
    After initialization, the diagnostic is called by using the
    `write` method.
    """

    def __init__(self, period, em, top, w3d, comm_world=None, 
                 fieldtypes=["rho", "E", "B", "J"], write_dir=None, 
                 lparallel_output=False) :
        """
        Initialize the field diagnostic.

        Parameters
        ----------
        period : int
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`)
            
        em : an EM3D object (as defined in em3dsolver)
            Contains the fields data and the different methods to extract it

        top : the object representing the `top` package in Warp
            Contains information on the time.

        w3d : the object representing the `w3d` package in Warp
            Contains the dimensions of the grid.

        comm_world : a communicator object
            Either an mpi4py or a pyMPI object, or None (single-proc)
            
        fieldtypes : a list of strings, optional
            The strings are either "rho", "E", "B" or "J"
            and indicate which field should be written.
            Default : all fields are written
            
        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        lparallel_output : boolean, optional
            Switch to set output mode (parallel or gathering)
            If "True" : Parallel output
        """
        # General setup
        OpenPMDDiagnostic.__init__(self, period, top, w3d, comm_world,
                                   lparallel_output, write_dir)
        
        # Register the arguments
        self.em = em
        self.fieldtypes = fieldtypes

    def setup_openpmd_meshespath( self, dset ) :
        """
        Set the attributes that are specific to the mesh path
        
        Parameter
        ---------
        dset : an h5py.Group object that contains all the mesh quantities
        """
        # Field Solver
        if self.em.stencil == 0 :
            dset.attrs["fieldSolver"] = "Yee"
            dset.attrs["fieldSolverOrder"] = 2
        elif self.em.stencil in [1,2] :
            dset.attrs["fieldSolver"] = "CK"
            dset.attrs["fieldSolverOrder"] = 2
        elif self.em.stencil == 3 :
            dset.attrs["fieldSolver"] = "Lehe"
            dset.attrs["fieldSolverOrder"] = 2
        # Current Smoothing
        if np.all( self.em.npass_smooth == 0 ) :
            dset.attrs["currentSmoothing"] = "none"
        else :
            dset.attrs["currentSmoothing"] = "digital"
            dset.attrs["currentSmoothingParameters"] = \
                str(self.em.npass_smooth)
        # Charge correction
        dset.attrs["chargeCorrection"] = "none"
        
    def setup_openpmd_meshrecord( self, dset ) :
        """
        Sets the attributes that are specific to a mesh record
        
        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object
        """
        # Geometry parameters
        if (self.em.l_2dxz==True) and (self.em.l_2drz==False) :
            self.geometry = "cartesian"
            self.gridSpacing = np.array([ self.em.dx, self.em.dz ])
        elif (self.em.l_2drz==True) :
            self.geometry = "thetaMode"
            self.geometryParameters = "m=%d;imag=+" %(self.em.circ_m + 1)
            self.gridSpacing = np.array([ self.em.dx, self.em.dz ])
        dset.attrs["gridUnitSI"] = 1.
        dset.attrs["gridGlobalOffset"] = \
          np.array([0., self.top.zgrid + self.w3d.zmmin])
        dset.attrs["dataOrder"] = "C"
        # Field Smoothing
        dset.attrs["fieldSmoothing"] = "none"

    def setup_openpmd_scalarrecord( self, dset, quantity ) :
        """
        Set up the attributes of a scalar record
    
        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object
        
        quantity : string
            The field that is being written
        """
        # Field positions
        positions = np.array([0., 0.])
        if quantity in ["rho", "Er", "Ex", "Et", "Ey",
                        "Jr", "Jx", "Jt", "Jy", "Bz" ] :
            # These fields are centered along the longitudinal direction
            positions[1] = 0. 
        elif quantity in ["Ez", "Jz", "Br", "Bx", "Bt", "By" ] :
            # These fields are staggered along the longitudinal direction
            positions[1] = 0.5
        else :
            raise ValueError("Unknown field quantity: %s" %quantity)
        if quantity in ["rho", "Et", "Ey", "Ez", "Jt", "Jy", "Jz", "Br", "Bx"] :
            # These fields are centered along the transverse direction
            positions[0] = 0.
        elif quantity in ["Er", "Ex", "Jr", "Jx", "Bt", "By", "Bz" ] :
            # These fields are staggered along the transverse direction
            positions[0] = 0.5
        else :
            raise ValueError("Unknown field quantity: %s" %quantity)
        dset.attrs["position"] = positions[:]


    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Create the file

        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "diags/hdf5", filename )

        # In gathering mode, only the first proc creates the file.
        if self.lparallel_output == False and self.rank == 0 :
            # Create the filename and open hdf5 file
            f = h5py.File( fullpath, mode="a" )
            self.setup_openpmd_file( f )
            this_rank_writes = True
        # In parallel mode (lparallel_output=True), all proc create the file
        elif self.lparallel_output == True :
            # Create the filename and open hdf5 file
            f = h5py.File( fullpath, mode="a", driver='mpio',
                           comm=self.comm_world)
            self.setup_openpmd_file( f )
            this_rank_writes = True
        else:
            f = None
            this_rank_writes = True

        # Setup the fields group
        if this_rank_writes :
            f.require_group("/fields")
            self.setup_openpmd_meshespath(f["/fields"])

        # Determine the components to be written (Cartesian or cylindrical)
        if (self.em.l_2drz == True) :
            coords = ['r', 't', 'z']
        else :
            coords = ['x', 'y', 'z']
            
        # Loop over the different quantities that should be written
        for fieldtype in self.fieldtypes :
            # Scalar field
            if fieldtype == "rho" :
                self.write_dataset( f, "/fields/rho", "rho", this_rank_writes )
                if this_rank_writes :
                    self.setup_openpmd_meshrecord( f["/fields/rho"] )
            # Vector field
            elif fieldtype in ["E", "B", "J"] :
                for coord in coords :
                    quantity = "%s%s" %(fieldtype, coord)
                    path = "/fields/%s/%s" %(fieldtype, coord)
                    self.write_dataset( f, path, quantity, this_rank_writes )
                if this_rank_writes :
                    self.setup_openpmd_meshrecord( f["/fields/%s" %fieldtype] )
            else :
                raise ValueError("Invalid string in fieldtypes: %s" %fieldtype)
        
        # Close the file
        if this_rank_writes :      
            f.close()

    def write_dataset( self, f, path, quantity, this_rank_writes ) :
        """
        Write a given dataset
    
        Parameters
        ----------
        f : an h5py.File object
    
        path : string
            The path where to write the dataset, inside the file f

        quantity : string
            Describes which field is being written.
            (Either rho, Er, Et, Ez, Br, Bz, Bt, Jr, Jt or Jz)

        this_rank_writes : bool
            Wether this proc participates in creating the dataset
            Parallel mode (lparallel_output=True) : all proc write
            Gathering mode : only the first proc writes
        """
        # 2D Cartesian case
        if (self.em.l_2dxz==True) and (self.em.l_2drz==False) :
            self.write_cart_dataset( f, path, quantity, this_rank_writes )
        # Circ case
        if (self.em.l_2drz==True) :
            self.write_circ_dataset( f, path, quantity, this_rank_writes )

        
    def write_circ_dataset( self, f, path, quantity, this_rank_writes ) :
        """
        Write a dataset in Circ coordinates
        
        See the docstring of write_dataset for the parameters
        """
        # Create the dataset and setup its attributes
        if this_rank_writes :
            # Shape of the data : first write the real part mode 0
            # and then the imaginary part of the mode 1
            datashape = (3, self.em.nx+1, self.em.nz+1)
            dset = f.require_dataset( path, datashape, dtype='f' )
            self.setup_openpmd_scalarrecord( dset, quantity )
            
        # Fill the dataset with these quantities
        # Gathering mode
        if self.lparallel_output == False :
            F, F_circ, _ = self.get_circ_dataset( quantity, lgather=True )
            if self.rank == 0:
	            # Mode m=0
    	        dset[0,:,:] = F
    	        if F_circ is not None:
        	        # Mode m=1 (real and imaginary part)
            	    dset[1,:,:] = F_circ[:,:,0].real
            	    dset[2,:,:] = F_circ[:,:,0].imag
        # Parallel mode
        else:
            F, F_circ, bounds = self.get_circ_dataset( quantity, False )
            # Mode m=0
            dset[ 0, bounds[0,0]:bounds[1,0],
                     bounds[0,1]:bounds[1,1] ] = F
            if F_circ is not None:
                # Mode m=1 (real and imaginary part)
                dset[ 1, bounds[0,0]:bounds[1,0],
                         bounds[0,1]:bounds[1,1] ] = F_circ[:,:,0].real
                dset[ 2, bounds[0,0]:bounds[1,0],
                         bounds[0,1]:bounds[1,1] ] = F_circ[:,:,0].imag


    def write_cart_dataset( self, f, path, quantity, this_rank_writes ) :
        """
        Write a dataset in Cartesian coordinates
        
        See the docstring of write_dataset for the parameters
        """
        # Create the dataset and setup its attributes
        if this_rank_writes :
            # Shape of the data : first write the real part mode 0
            # and then the imaginary part of the mode 1
            datashape = (self.em.nx+1, self.em.nz+1)
            dset = f.require_dataset( path, datashape, dtype='f' )
            self.setup_openpmd_scalarrecord( dset, quantity )
            
        # Fill the dataset with these quantities
        # Gathering mode
        if self.lparallel_output == False :
            F, _ = self.get_cart_dataset( quantity, True )
            if self.rank == 0:
    	        dset[:,:] = F
        # Parallel mode
        else:
            F, bounds = self.get_cart_dataset( quantity, False )
            dset[ 0, bounds[0,0]:bounds[1,0],
                     bounds[0,1]:bounds[1,1] ] = F
                     
    def get_circ_dataset( self, quantity, lgather) :
        """
        Get a given quantity in Circ coordinates

        Parameters
        ----------
        quantity : string
            Describes which field is being written.
            (Either rho, Er, Et, Ez, Br, Bz, Bt, Jr, Jt or Jz)

        lgather : boolean
            Defines if data is gathered on me (process) = 0
            If "False": No gathering is done
        """
        F_circ = None
        em = self.em
        
        # Treat the currents in a special way
        if quantity in ['Jr', 'Jt', 'Jz'] :
            # Get the array index that corresponds to that component
            i = circ_dict_Jindex[ quantity ]
            # Extract mode 0
            F = em.getarray( em.fields.J[:,:,:,i] )
            # Extract higher modes
            if em.circ_m > 0 :
                F_circ = em.getarray_circ( em.fields.J_circ[:,:,i,:] )
                
        # Treat the fields E, B, rho in a more systematic way
        elif quantity in ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz', 'rho' ] :
            # Get the field name in Warp
            field_name = circ_dict_quantity[ quantity ]
            # Extract mode 0
            field_array = getattr( em.fields, field_name )
            F = em.getarray( field_array )
            # Extract higher modes
            if em.circ_m > 0 :
                field_array = getattr( em.fields, field_name + '_circ')
                F_circ = em.getarray_circ( field_array )

        # Gather array if lgather = True 
        # (Mutli-proc operations using gather)
        # Only done in non-parallel case
        if lgather == True:
            F = em.gatherarray( F )
            if em.circ_m > 0 :
                F_circ = em.gatherarray( F_circ )
        
        # Get global positions (indices) of local domain
        # Only needed for parallel output
        if lgather == False :
            nx, nz = np.shape(F)
            bounds = np.zeros([2,2], dtype = np.int)
            bounds[0,0] = int((em.block.xmin - em.xmmin) / em.dx)
            bounds[1,0] = bounds[0,1] + nx
            bounds[0,1] = int((em.block.zmin - em.zmmin) / em.dz)
            bounds[1,1] = bounds[0,0] + nz
        else :
        	bounds = None

        return( F, F_circ, bounds )

        
    def get_cart_dataset( self, quantity, lgather) :
        """
        Get a given quantity in Cartisian coordinates

        Parameters
        ----------
        quantity : string
            Describes which field is being written.
            (Either rho, Er, Et, Ez, Br, Bz, Bt, Jr, Jt or Jz)

        lgather : boolean
            Defines if data is gathered on me (process) = 0
            If "False": No gathering is done
        """
        em = self.em
        
        # Treat the currents in a special way
        if quantity in ['Jx', 'Jy', 'Jz'] :
            # Get the array index that corresponds to that component
            i = cart_dict_Jindex[ quantity ]
            F = em.getarray( em.fields.J[:,:,:,i] )

        # Treat the fields E, B, rho in a more systematic way
        elif quantity in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho' ] :
            # Get the field name in Warp
            field_name = cart_dict_quantity[ quantity ]
            field_array = getattr( em.fields, field_name )
            F = em.getarray( field_array )

        # Gather array if lgather = True 
        # (Mutli-proc operations using gather)
        # Only done in non-parallel case
        if lgather == True:
            F = em.gatherarray( F )
        
        # Get global positions (indices) of local domain
        # Only needed for parallel output
        if lgather == False :
            nx, nz = np.shape(F)
            bounds = np.zeros([2,2], dtype = np.int)
            bounds[0,0] = int((em.block.xmin - em.xmmin) / em.dx)
            bounds[1,0] = bounds[0,1] + nx
            bounds[0,1] = int((em.block.zmin - em.zmmin) / em.dz)
            bounds[1,1] = bounds[0,0] + nz
        else :
        	bounds = None

        return( F, bounds )
