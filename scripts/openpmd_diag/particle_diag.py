"""
This file defines the class ParticleDiagnostic
"""
import os
import h5py
import numpy as np
from scipy import constants
from generic_diag import OpenPMDDiagnostic
from parallel import gatherarray
from data_dict import macro_weighted_dict, weighting_power_dict

class ParticleDiagnostic(OpenPMDDiagnostic) :
    """
    Class that defines the particle diagnostics to be done.

    Usage
    -----
    After initialization, the diagnostic is called by using the
    `write` method.
    """

    def __init__(self, period, top, w3d, comm_world=None,
                 species = {"electrons": None},
                 particle_data=["position", "momentum", "weighting"],
                 select=None, write_dir=None, lparallel_output=False) :
        """
        Initialize the field diagnostics.

        Parameters
        ----------
        period : int
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`)

        top : the object representing the `top` package in Warp
            Contains information on the time.

        w3d : the object representing the `w3d` package in Warp
            Contains the dimensions of the grid.

        comm_world : a communicator object
            Either an mpi4py or a pyMPI object, or None (single-proc)
        
        species : a dictionary of Species objects
            The Species object that is written (e.g. elec)
            is assigned to the particleName of this species.
            (e.g. "electrons")

        particle_data : a list of strings, optional 
            The particle properties are given by:
            ["position", "momentum", "weighting"]
            for the coordinates x,y,z.
            Default : electron particle data is written

        select : dict, optional
            Either None or a dictionary of rules
            to select the particles, of the form
            'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
            'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
            'uz' : [5., None]  (Particles with uz above 5 mc)
            
        write_dir : a list of strings, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        lparallel_output : boolean
            Switch to set output mode (parallel or gathering)
            If "True" : Parallel output
        """
        # General setup
        OpenPMDDiagnostic.__init__(self, period, top, w3d, comm_world,
                                   lparallel_output, write_dir)
        
        # Register the arguments
        self.particle_data = particle_data
        self.species_dict = species
        self.select = select

        # Correct the bounds in momenta (since the momenta in Warp
        # are not unitless, but have the units of a velocity)
        for momentum in ['ux', 'uy', 'uz']:
            if (self.select is not None) and (momentum in self.select):
                if self.select[momentum][0] is not None:
                    self.select[momentum][0] *= constants.c
                if self.select[momentum][1] is not None:
                    self.select[momentum][1] *= constants.c
 
    def setup_openpmd_species_group( self, grp, species ) :
        """
        Set the attributes that are specific to the particle group
        
        Parameter
        ---------
        grp : an h5py.Group object
            Contains all the species
    
        species : a Warp species.Species object
        """
        # Generic attributes
        grp.attrs["particleShape"] = float( self.top.depos_order[0][0] )
        grp.attrs["currentDeposition"] = np.string_("Esirkepov")
        grp.attrs["particleSmoothing"] = np.string_("none")
        # Particle pusher
        if self.top.pgroup.lebcancel_pusher==True :
            grp.attrs["particlePush"] = np.string_("Vay")
        else :
            grp.attrs["particlePush"] = np.string_("Boris")
        # Particle shape
        if np.all( self.top.efetch==1 ) :
            grp.attrs["particleInterpolation"]= np.string_("momentumConserving")
        elif np.all( self.top.efetch==4 ) :
            grp.attrs["particleInterpolation"]= np.string_("energyConserving")
        
        # Setup constant datasets
        for quantity in ["charge", "mass", "positionOffset"] :
            grp.require_group(quantity)
            self.setup_openpmd_species_record( grp[quantity], quantity )
        for quantity in ["charge", "mass", "positionOffset/x",
                            "positionOffset/y", "positionOffset/z"] :
            grp.require_group(quantity)
            self.setup_openpmd_species_component( grp[quantity], quantity )
            grp[quantity].attrs["shape"] = np.array([1], dtype=np.uint64)
            # Required. Since it is not really used, the shape is 1 here.
        # Set the corresponding values
        grp["charge"].attrs["value"] = species.charge
        grp["mass"].attrs["value"] = species.mass
        grp["positionOffset/x"].attrs["value"] = 0.
        grp["positionOffset/y"].attrs["value"] = 0.
        grp["positionOffset/z"].attrs["value"] = 0.

    def setup_openpmd_species_record( self, grp, quantity ) :
        """
        Set the attributes that are specific to a species record
        
        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset
            The group that correspond to `quantity`
            (in particular, its path must end with "/<quantity>")
    
        quantity : string
            The name of the record being setup
            e.g. "position", "momentum"
        """
        # Generic setup
        self.setup_openpmd_record( grp, quantity )

        # Weighting information
        grp.attrs["macroWeighted"] = macro_weighted_dict[quantity]
        grp.attrs["weightingPower"] = weighting_power_dict[quantity]

    def setup_openpmd_species_component( self, grp, quantity ) :
        """
        Set the attributes that are specific to a species component

        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset
        
        quantity : string
            The name of the component
        """
        self.setup_openpmd_component( grp )
        
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
            this_rank_writes = False

        # Loop over the different species and 
        # particle quantities that should be written
        for species_name in self.species_dict :

            # Check if the species exists
            species = self.species_dict[species_name]
            if species is None :
                # If not, immediately go to the next species_name
                continue

            # Setup the species group
            if this_rank_writes :
                species_path = "/data/%d/particles/%s" %(iteration,
                                                         species_name)
                # Create and setup the h5py.Group species_grp
                species_grp = f.require_group( species_path )
                self.setup_openpmd_species_group( species_grp, species )
            else:
                species_grp = None
                
            # Select the particles that will be written
            select_array = self.apply_selection( species )
            # Get their total number
            n = select_array.sum()
            if self.comm_world is not None :
            	# In MPI mode: gather and broadcast an array containing 
            	# the number of particles on each process 
                n_rank = self.comm_world.allgather(n)
                N = sum(n_rank)
            else:
            	# Single-proc output
                n_rank = None
                N = n

            # Write the datasets for each particle datatype
            self.write_particles( species_grp, species,
                         n_rank, N, select_array, this_rank_writes )
        
        # Close the file
        if self.lparallel_output == True or self.rank == 0 :      
            f.close()

    def write_particles( self, species_grp, species,
                         n_rank, N, select_array, this_rank_writes ) :
        """
        Write all the particle data sets for one given species
    
        species_grp : an h5py.Group
            The group where to write the species considered

        species : a warp Species object
        	The species object to get the particle data from 

        n_rank: an array with dtype = int of size = n_procs
        	Contains the local number of particles for each process
            
        N : int
        	Contains the global number of particles

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select

        this_rank_writes : bool
            Determines whether the present rank contributes in writing the data
        """
        for particle_var in self.particle_data :
                
            if particle_var == "position" :
                for coord in ["x", "y", "z"] :
                    quantity = coord
                    quantity_path = "%s/%s" %(particle_var, coord)
                    self.write_dataset( species_grp, species, quantity_path,
                        quantity, n_rank, N, select_array )
                if this_rank_writes :
                    self.setup_openpmd_species_record(
                        species_grp[particle_var], particle_var )
                        
            elif particle_var == "momentum" :
                for coord in ["x", "y", "z"] :
                    quantity = "u%s" %(coord)
                    quantity_path = "%s/%s" %(particle_var, coord)
                    self.write_dataset( species_grp, species, quantity_path,
                                        quantity, n_rank, N, select_array )
                if this_rank_writes :
                    self.setup_openpmd_species_record( 
                        species_grp[particle_var], particle_var )
                        
            elif particle_var == "weighting" :
                quantity = "w"
                quantity_path = "weighting"
                self.write_dataset( species_grp, species, quantity_path,
                                    quantity, n_rank, N, select_array )
                if this_rank_writes :
                    self.setup_openpmd_species_record( 
                        species_grp[particle_var], particle_var )
                
            else :
                raise ValueError("Invalid string in %s of species" 
                    				 %(particle_var))

    def apply_selection( self, species ) :
        """
        Apply the rules of self.select to determine which
        particles should be written

        Parameters
        ----------
        species : a Species object

        Returns
        -------
        A 1d array of the same shape as that particle array
        containing True for the particles that satify all
        the rules of self.select
        """
        # Initialize an array filled with True
        select_array = np.ones( species.getn(gather=0), dtype='bool' )

        # Apply the rules successively
        if self.select is not None :
            # Go through the quantities on which a rule applies
            for quantity in self.select.keys() :
                quantity_array = self.get_quantity( species, quantity )
                # Lower bound
                if self.select[quantity][0] is not None :
                    select_array = np.logical_and(
                        quantity_array > self.select[quantity][0],
                        select_array )
                # Upper bound
                if self.select[quantity][1] is not None :
                    select_array = np.logical_and(
                        quantity_array < self.select[quantity][1],
                        select_array )

        return( select_array )

        
    def write_dataset( self, species_grp, species, path, quantity,
                       n_rank, N, select_array ) :
        """
        Write a given dataset
    
        Parameters
        ----------
        species_grp : an h5py.Group
            The group where to write the species considered
        
        species : a warp Species object
        	The species object to get the particle data from

        path : string
            The relative path where to write the dataset, inside the species_grp

        quantity : string
            Describes which quantity is written
            x, y, z, ux, uy, uz, w
            
        n_rank: an array with dtype = int of size = n_procs
        	Contains the local number of particles for each process
            
        N : int
        	Contains the global number of particles

        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        """
        # Create the dataset and setup its attributes
        if self.lparallel_output == True or self.rank == 0 :
            datashape = (N, )
            dset = species_grp.require_dataset( path, datashape, dtype='f')
            self.setup_openpmd_species_component( dset, quantity )
        else :
            dset = None
            
        # Fill the dataset with the quantity
        # (Single-proc operation, when using gathering)
        if self.lparallel_output == False :
            quantity_array = self.get_dataset( species,
                    quantity, select_array, gather=True )
            if self.rank == 0:
                dset[:] = quantity_array
        # Fill the dataset with these quantities with respect
        # to the global position of the local domain
        # (truly parallel HDF5 output)
        else :
            quantity_array = self.get_dataset( species, 
                    quantity, select_array, gather=False )
            # Calculate last index occupied by previous rank
            nold = sum(n_rank[0:self.rank])
            # Calculate the last index occupied by the current rank
            nnew = nold+n_rank[self.rank]
            # Write the local data to the global array
            dset[nold:nnew] = quantity_array
            
    def get_dataset( self, species, quantity, select_array, gather ) :
        """
        Extract the array that satisfies select_array
        
        species : a Particles object
        	The species object to get the particle data from 

        quantity : string
            The quantity to be extracted (e.g. 'x', 'uz', 'w')
            
        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select

        gather : bool
            Whether to gather the fields on the first processor
        """

        # Extract the quantity
        quantity_array = self.get_quantity( species, quantity )
        
        # Apply the selection
        quantity_array = quantity_array[ select_array ]

        # If this is the momentum, mutliply by the proper factor
        if quantity in ['ux', 'uy', 'uz']:
            quantity_array *= species.mass
        
        # Gather the data if required
        if gather==False :
            return( quantity_array )
        else :
            return(gatherarray( quantity_array, root=0, comm=self.comm_world ))

            
    def get_quantity( self, species, quantity ) :
        """
        Get a given quantity

        Parameters
        ----------
        species : a Species object
            Contains the species object to get the particle data from

        quantity : string
            Describes which quantity is queried
            Either "x", "y", "z", "ux", "uy", "uz" or "w"
        """
        # Extract the chosen quantities

        if quantity == "x" :
            quantity_array = species.getx(gather=False)
        elif quantity == "y" :
            quantity_array = species.gety(gather=False)
        elif quantity == "z" :
            quantity_array = species.getz(gather=False)
        elif quantity == "ux" :
            quantity_array = species.getux(gather=False)
        elif quantity == "uy" :
            quantity_array = species.getuy(gather=False)
        elif quantity == "uz" :
            quantity_array = species.getuz(gather=False)
        elif quantity == "w" :
            quantity_array = species.getweights(gather=False)

        return( quantity_array )

        
